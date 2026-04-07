"""
Decentralized Federated Learning (DFL) simulation.

Simulates gossip-based decentralized SGD on a single GPU. All n node models
are stored as a flat parameter tensor [n, P]. Gradients are computed via
functional API to avoid costly unflatten/flatten per node.

Usage:
    python train.py --topo qrsdr --n 32 --d 4 --alpha 0.1 --seed 0
    python train.py --topo base --n 32 --d 1 --alpha 0.1 --rounds 2000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import create_model, count_parameters
from data import (load_dataset, partition_iid, partition_dirichlet,
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


def get_param_shapes(model):
    """Get list of (shape, numel) for each parameter."""
    return [(p.shape, p.numel()) for p in model.parameters()]


def compute_grad_for_node(model, param_shapes, flat_params, images, labels):
    """
    Compute gradient for one node using functional approach.
    Avoids full unflatten/flatten — only sets params once, computes grad once.
    Returns flat gradient tensor.
    """
    # Set model params from flat vector
    offset = 0
    for p, (shape, numel) in zip(model.parameters(), param_shapes):
        p.data.copy_(flat_params[offset:offset + numel].view(shape))
        offset += numel

    # Forward + backward
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Extract flat gradient
    grads = []
    for p in model.parameters():
        grads.append(p.grad.view(-1))
    return torch.cat(grads)


def evaluate(model, param_shapes, flat_params, test_loader, device):
    """Evaluate the global (averaged) model on the test set."""
    offset = 0
    for p, (shape, numel) in zip(model.parameters(), param_shapes):
        p.data.copy_(flat_params[offset:offset + numel].view(shape))
        offset += numel
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
    train_set, test_set, num_classes = load_dataset(args.dataset, args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2)

    if args.alpha is None or args.alpha >= 100:
        partition = partition_iid(train_set, args.n, seed=args.seed)
        data_label = "iid"
    else:
        partition = partition_dirichlet(train_set, args.n, args.alpha,
                                        seed=args.seed)
        data_label = f"dir{args.alpha}"

    print(f"\nData: {args.dataset}, {data_label}, {args.n} nodes")
    partition_stats(train_set, partition)

    node_loaders = create_data_loaders(train_set, partition,
                                        batch_size=args.batch_size)
    node_iters = [iter(loader) for loader in node_loaders]

    def get_batch(node_id):
        nonlocal node_iters
        try:
            return next(node_iters[node_id])
        except StopIteration:
            node_iters[node_id] = iter(node_loaders[node_id])
            return next(node_iters[node_id])

    # --- Topology ---
    topo_seed = args.topo_seed if args.topo_seed >= 0 else args.seed
    topo = get_topology(args.topo, args.n, d=args.d, seed=topo_seed)

    W_list = [W.to(device) for W in topo.W_list]
    meta = topo.meta

    edges_str = f"edges={meta['edges']}"
    degree_str = f"d={meta.get('degree', '?')}"
    print(f"\nTopology: {args.topo}, n={args.n}, {degree_str}, {edges_str}")
    print(f"  lambda_2 = {meta['lambda2']:.6f}")
    print(f"  spectral_gap = {meta['spectral_gap']:.6f}")
    if topo.time_varying:
        print(f"  time-varying: {len(W_list)} rounds in sequence")
        if 'ftc_verified' in meta:
            print(f"  finite-time consensus: {meta['ftc_verified']}")

    # --- Model ---
    model = create_model(num_classes=num_classes, device=device)
    P = count_parameters(model)
    param_shapes = get_param_shapes(model)
    print(f"\nModel: ResNet-20, {P:,} parameters ({P * 4 / 1e6:.2f} MB)")
    print(f"Total memory: {args.n} nodes x {P * 4 / 1e6:.2f} MB = "
          f"{args.n * P * 4 / 1e6:.1f} MB")

    # Initialize all nodes to the same weights
    init_flat = flatten_params(model).clone()
    node_params = init_flat.unsqueeze(0).expand(args.n, -1).clone()

    # --- Optimizer state: per-node momentum buffers ---
    momentum = torch.zeros_like(node_params)

    # --- Pre-fetch and stack all node batches to GPU at once ---
    # Instead of moving one batch at a time, we gather all node batches
    # per local step and move them in bulk

    # --- LR schedule ---
    total_steps = args.rounds * args.tau
    step_count = 0

    def get_lr(step):
        return args.lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))

    # --- Training loop ---
    log = []
    t0 = time.time()
    print(f"\nTraining: {args.rounds} rounds x tau={args.tau} local steps "
          f"= {total_steps} total SGD steps")
    print(f"Eval every {args.eval_freq} rounds")
    print("=" * 70)

    num_W = len(W_list)

    # Pre-allocate gradient buffer
    grad_buffer = torch.zeros(P, device=device)

    for t in range(1, args.rounds + 1):
        # 1. tau local SGD steps per node
        for _local in range(args.tau):
            lr = get_lr(step_count)
            step_count += 1

            # Pre-fetch ALL node batches to GPU first
            all_images = []
            all_labels = []
            for i in range(args.n):
                images, labels = get_batch(i)
                all_images.append(images)
                all_labels.append(labels)

            # Stack into mega-batches (move to GPU once)
            # Each node has potentially different batch size, so we
            # process them sequentially but with data already on GPU
            for i in range(args.n):
                images = all_images[i].to(device, non_blocking=True)
                labels = all_labels[i].to(device, non_blocking=True)

                # Set model params (fast: just copy from flat tensor)
                offset = 0
                for p, (shape, numel) in zip(model.parameters(), param_shapes):
                    p.data.copy_(node_params[i, offset:offset + numel].view(shape))
                    offset += numel

                model.train()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                model.zero_grad()
                loss.backward()

                # Read gradient directly into momentum update (no cat needed)
                offset = 0
                for p, (shape, numel) in zip(model.parameters(), param_shapes):
                    g = p.grad.view(-1)
                    momentum[i, offset:offset + numel].mul_(args.momentum).add_(g)
                    node_params[i, offset:offset + numel].add_(
                        momentum[i, offset:offset + numel] +
                        args.weight_decay * node_params[i, offset:offset + numel],
                        alpha=-lr)
                    offset += numel

        # 2. Gossip averaging: params = W @ params
        W = W_list[(t - 1) % num_W]
        node_params = W @ node_params

        # 3. Evaluate
        if t % args.eval_freq == 0 or t == 1:
            global_params = node_params.mean(dim=0)
            acc = evaluate(model, param_shapes, global_params,
                          test_loader, device)

            elapsed = time.time() - t0
            rounds_per_sec = t / elapsed if elapsed > 0 else 0

            entry = {
                "round": t,
                "accuracy": acc,
                "lr": lr,
                "elapsed": elapsed,
            }
            log.append(entry)

            cur_lr = get_lr(step_count - 1)
            print(f"  round {t:5d}/{args.rounds} | acc {acc:.4f} | "
                  f"lr {cur_lr:.5f} | {rounds_per_sec:.1f} r/s | "
                  f"{elapsed:.0f}s")

    # --- Save results ---
    os.makedirs(args.output_dir, exist_ok=True)

    result = {
        "args": vars(args),
        "topology": {
            "name": args.topo,
            "n": args.n,
            "d": args.d,
            "lambda2": meta["lambda2"],
            "spectral_gap": meta["spectral_gap"],
            "edges": meta["edges"],
            "time_varying": topo.time_varying,
            "num_rounds_in_sequence": len(W_list),
        },
        "log": log,
    }

    fname = (f"{args.dataset}_{args.topo}_n{args.n}_d{args.d}_{data_label}"
             f"_s{args.seed}.json")
    path = os.path.join(args.output_dir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="DFL Benchmark")
    parser.add_argument("--topo", type=str, required=True,
                        choices=TOPOLOGY_NAMES)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, default=4,
                        help="Degree (for qrsdr/random) or k (for base)")
    parser.add_argument("--topo-seed", type=int, default=-1)

    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="./data")

    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
