"""
Decentralized Federated Learning (DFL) simulation.

Simulates gossip-based decentralized SGD on a single GPU. All n node models
are stacked as a parameter tensor [n, P], and gossip averaging is a single
matmul W @ params.

Usage:
    python train.py --topo er --n 32 --m 64 --alpha 0.1 --seed 0
    python train.py --topo ring --n 128 --alpha 1.0 --rounds 3000

Reference for D-PSGD:
    Lian et al., "Can Decentralized Algorithms Outperform Centralized Algorithms?
    A Case Study for Decentralized Parallel Stochastic Gradient Descent",
    NeurIPS 2017.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import create_model, count_parameters
from data import (load_cifar10, partition_iid, partition_dirichlet,
                  create_data_loaders, partition_stats)
from topology import get_topology, TOPOLOGY_NAMES


def flatten_params(model):
    """Flatten all model parameters into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def unflatten_params(model, flat):
    """Load a flat parameter vector back into a model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view(p.shape))
        offset += numel


def evaluate(model, flat_params, test_loader, device):
    """Evaluate the global (averaged) model on the test set."""
    unflatten_params(model, flat_params)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # --- Data ---
    train_set, test_set = load_cifar10(args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2)

    if args.alpha is None or args.alpha >= 100:
        partition = partition_iid(train_set, args.n, seed=args.seed)
        data_label = "iid"
    else:
        partition = partition_dirichlet(train_set, args.n, args.alpha,
                                        seed=args.seed)
        data_label = f"dir{args.alpha}"

    print(f"\nData: {data_label}, {args.n} nodes")
    partition_stats(train_set, partition)

    node_loaders = create_data_loaders(train_set, partition,
                                        batch_size=args.batch_size)
    # Infinite iterators per node
    node_iters = [iter(loader) for loader in node_loaders]

    def get_batch(node_id):
        nonlocal node_iters
        try:
            return next(node_iters[node_id])
        except StopIteration:
            node_iters[node_id] = iter(node_loaders[node_id])
            return next(node_iters[node_id])

    # --- Topology ---
    # Ring doesn't use m; others load from adj binaries
    topo_seed = args.topo_seed if args.topo_seed >= 0 else args.seed
    if args.topo == "ring":
        _, W, meta = get_topology("ring", args.n, 0, 0, args.data_dir)
    else:
        _, W, meta = get_topology(args.topo, args.n, args.m, topo_seed,
                                  args.data_dir)

    W = W.to(device)
    print(f"\nTopology: {args.topo}, n={args.n}, m={meta['edges']}")
    print(f"  lambda_2 = {meta['lambda2']:.6f}")
    print(f"  spectral_gap = {meta['spectral_gap']:.6f}")

    # --- Model ---
    model = create_model(device)
    P = count_parameters(model)
    print(f"\nModel: ResNet-20, {P:,} parameters ({P * 4 / 1e6:.2f} MB)")
    print(f"Total memory: {args.n} nodes x {P * 4 / 1e6:.2f} MB = "
          f"{args.n * P * 4 / 1e6:.1f} MB")

    # Initialize all nodes to the same weights
    init_flat = flatten_params(model).clone()
    # node_params: [n, P] — each row is one node's flattened parameters
    node_params = init_flat.unsqueeze(0).expand(args.n, -1).clone()

    # --- Optimizer state: per-node momentum buffers ---
    momentum = torch.zeros_like(node_params)

    # --- LR schedule: cosine decay ---
    def get_lr(t):
        return args.lr * 0.5 * (1 + np.cos(np.pi * t / args.rounds))

    # --- Training loop ---
    log = []
    t0 = time.time()
    print(f"\nTraining: {args.rounds} rounds, eval every {args.eval_freq}")
    print("=" * 70)

    for t in range(1, args.rounds + 1):
        lr = get_lr(t)

        # 1. Local SGD step for each node
        for i in range(args.n):
            images, labels = get_batch(i)
            images, labels = images.to(device), labels.to(device)

            # Load node params into model
            unflatten_params(model, node_params[i])
            model.train()

            # Forward + backward
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()

            # SGD with momentum: v = mu*v + grad; w = w - lr*v
            grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
            momentum[i] = args.momentum * momentum[i] + grad
            node_params[i] -= lr * (momentum[i] + args.weight_decay * node_params[i])

        # 2. Gossip averaging: params = W @ params
        node_params = W @ node_params

        # 3. Evaluate
        if t % args.eval_freq == 0 or t == 1:
            # Global model = mean of all node params
            global_params = node_params.mean(dim=0)
            acc = evaluate(model, global_params, test_loader, device)

            elapsed = time.time() - t0
            rounds_per_sec = t / elapsed if elapsed > 0 else 0

            entry = {
                "round": t,
                "accuracy": acc,
                "lr": lr,
                "elapsed": elapsed,
            }
            log.append(entry)

            print(f"  round {t:5d}/{args.rounds} | acc {acc:.4f} | "
                  f"lr {lr:.5f} | {rounds_per_sec:.1f} r/s | "
                  f"{elapsed:.0f}s")

    # --- Save results ---
    os.makedirs(args.output_dir, exist_ok=True)

    result = {
        "args": vars(args),
        "topology": {
            "name": args.topo,
            "n": args.n,
            "m": meta["edges"],
            "lambda2": meta["lambda2"],
            "spectral_gap": meta["spectral_gap"],
        },
        "log": log,
    }

    fname = (f"{args.topo}_n{args.n}_m{args.m}_{data_label}"
             f"_s{args.seed}.json")
    path = os.path.join(args.output_dir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="DFL Benchmark")
    # Topology
    parser.add_argument("--topo", type=str, required=True,
                        choices=TOPOLOGY_NAMES,
                        help="Topology name")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of nodes")
    parser.add_argument("--m", type=int, default=0,
                        help="Number of edges (ignored for ring)")
    parser.add_argument("--topo-seed", type=int, default=-1,
                        help="Topology seed index (default: same as --seed)")

    # Data
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet alpha (None=IID, 0.1=severe non-IID)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="./data")

    # Training
    parser.add_argument("--rounds", type=int, default=2000,
                        help="Total communication rounds")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    # Ring has fixed m = n
    if args.topo == "ring":
        args.m = args.n

    train(args)


if __name__ == "__main__":
    main()
