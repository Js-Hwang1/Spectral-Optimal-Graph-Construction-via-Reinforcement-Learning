"""
Decentralized Federated Learning (DFL) simulation.

Simulates gossip-based decentralized SGD on a single GPU.
Uses torch.vmap to vectorize forward+backward across all n nodes
in parallel — no sequential per-node loop.

Usage:
    python train.py --topo ours --n 32 --d 4 --alpha 0.1 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

from models import create_model, count_parameters
from data import (load_dataset, partition_iid, partition_dirichlet,
                  create_data_loaders, partition_stats)
from topology import get_topology, TOPOLOGY_NAMES


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def unflatten_to_dict(model, flat):
    """Convert flat vector to a {name: tensor} dict matching model params."""
    param_dict = {}
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        param_dict[name] = flat[offset:offset + numel].view(p.shape)
        offset += numel
    return param_dict


def evaluate(model, flat_params, test_loader, device):
    param_dict = unflatten_to_dict(model, flat_params)
    # Load into model for eval
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(param_dict[name])
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

    print(f"\nTopology: {args.topo}, n={args.n}, d={meta.get('degree', '?')}, "
          f"edges={meta['edges']}")
    print(f"  lambda_2 = {meta['lambda2']:.6f}")
    print(f"  spectral_gap = {meta['spectral_gap']:.6f}")
    if topo.time_varying:
        print(f"  time-varying: {len(W_list)} rounds in sequence")
        if 'ftc_verified' in meta:
            print(f"  finite-time consensus: {meta['ftc_verified']}")

    # --- Model ---
    model = create_model(num_classes=num_classes, device=device)
    P = count_parameters(model)
    print(f"\nModel: ResNet-20, {P:,} parameters ({P * 4 / 1e6:.2f} MB)")
    print(f"Total memory: {args.n} nodes x {P * 4 / 1e6:.2f} MB = "
          f"{args.n * P * 4 / 1e6:.1f} MB")

    # --- Build vectorized gradient function ---
    # functional_call lets us call model with arbitrary param dicts
    # vmap parallelizes across the n-node dimension

    # Get base param dict structure (for shape reference)
    base_params = {name: p.detach() for name, p in model.named_parameters()}
    param_names = list(base_params.keys())
    param_shapes = [(name, base_params[name].shape, base_params[name].numel())
                    for name in param_names]

    # Set model buffers (BatchNorm running stats) to not track
    # since we can't vmap through running stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    def compute_loss_single(flat_params, images, labels):
        """Compute loss for a single node given flat params and a batch."""
        param_dict = {}
        offset = 0
        for name, shape, numel in param_shapes:
            param_dict[name] = flat_params[offset:offset + numel].view(shape)
            offset += numel
        out = functional_call(model, param_dict, (images,))
        return F.cross_entropy(out, labels)

    # grad w.r.t. flat_params (arg 0)
    compute_grad_single = grad(compute_loss_single, argnums=0)

    # Try vmap: vectorize across node dimension
    # flat_params: [n, P], images: [n, B, C, H, W], labels: [n, B]
    try:
        compute_grad_batched = vmap(compute_grad_single, in_dims=(0, 0, 0))
        use_vmap = True
        print("\n  Using vmap for parallel node gradient computation")
    except Exception as e:
        use_vmap = False
        print(f"\n  vmap not available ({e}), falling back to sequential")

    # --- Init ---
    init_flat = flatten_params(model).clone()
    node_params = init_flat.unsqueeze(0).expand(args.n, -1).clone()
    momentum = torch.zeros_like(node_params)

    total_steps = args.rounds * args.tau
    step_count = 0

    def get_lr(step):
        return args.lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))

    # --- Training loop ---
    log = []
    t0 = time.time()
    num_W = len(W_list)
    print(f"\nTraining: {args.rounds} rounds x tau={args.tau} local steps "
          f"= {total_steps} total SGD steps")
    print(f"Eval every {args.eval_freq} rounds")
    print("=" * 70)

    for t in range(1, args.rounds + 1):
        for _local in range(args.tau):
            lr = get_lr(step_count)
            step_count += 1

            # Gather all node batches
            batch_images = []
            batch_labels = []
            for i in range(args.n):
                images, labels = get_batch(i)
                batch_images.append(images)
                batch_labels.append(labels)

            if use_vmap:
                # Pad batches to same size for stacking
                max_bs = max(img.shape[0] for img in batch_images)
                padded_images = []
                padded_labels = []
                for img, lbl in zip(batch_images, batch_labels):
                    bs = img.shape[0]
                    if bs < max_bs:
                        # Pad with repeated last sample
                        pad_img = img[-1:].expand(max_bs - bs, -1, -1, -1)
                        pad_lbl = lbl[-1:].expand(max_bs - bs)
                        img = torch.cat([img, pad_img], dim=0)
                        lbl = torch.cat([lbl, pad_lbl], dim=0)
                    padded_images.append(img)
                    padded_labels.append(lbl)

                # Stack: [n, B, C, H, W] and [n, B]
                stacked_images = torch.stack(padded_images).to(device)
                stacked_labels = torch.stack(padded_labels).to(device)

                # Compute all gradients in parallel
                all_grads = compute_grad_batched(
                    node_params, stacked_images, stacked_labels)

                # SGD + momentum update (vectorized)
                momentum.mul_(args.momentum).add_(all_grads)
                node_params.add_(
                    momentum + args.weight_decay * node_params,
                    alpha=-lr)
            else:
                # Sequential fallback (still optimized vs original)
                for i in range(args.n):
                    images = batch_images[i].to(device)
                    labels = batch_labels[i].to(device)

                    g = compute_grad_single(node_params[i], images, labels)

                    momentum[i].mul_(args.momentum).add_(g)
                    node_params[i].add_(
                        momentum[i] + args.weight_decay * node_params[i],
                        alpha=-lr)

        # Gossip averaging
        W = W_list[(t - 1) % num_W]
        node_params = W @ node_params

        # Evaluate
        if t % args.eval_freq == 0 or t == 1:
            global_params = node_params.mean(dim=0)
            acc = evaluate(model, global_params, test_loader, device)

            elapsed = time.time() - t0
            rps = t / elapsed if elapsed > 0 else 0

            entry = {"round": t, "accuracy": acc,
                     "lr": get_lr(step_count - 1), "elapsed": elapsed}
            log.append(entry)

            print(f"  round {t:5d}/{args.rounds} | acc {acc:.4f} | "
                  f"lr {entry['lr']:.5f} | {rps:.1f} r/s | "
                  f"{elapsed:.0f}s")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "args": vars(args),
        "topology": {
            "name": args.topo, "n": args.n, "d": args.d,
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
    parser.add_argument("--d", type=int, default=4)
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
