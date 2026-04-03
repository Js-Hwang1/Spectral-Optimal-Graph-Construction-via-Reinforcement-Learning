#!/usr/bin/env python3
"""
Collect FV expert decision traces for supervised pre-training.

Parallelized across (n, m, graph_idx) tasks using multiprocessing.

FV decision rule at node u:
  - Compute cut scores: (v₂[u] - v₂[i])² for all i
  - Among u's neighbors: the one with LOWEST cut score is the best to drop
  - Among u's non-neighbors: the one with HIGHEST cut score is the best to add
  - Accept swap only if it actually improves λ₂; otherwise KEEP

Output: .npz file with arrays:
  features:  (N_samples, max_n, 5)  — node feature matrices
  nb_actions: (N_samples,)          — expert neighbor action (index or n=KEEP)
  dst_actions:(N_samples,)          — expert destination action
  n_vals:    (N_samples,)           — graph size for this sample
  nb_masks:  (N_samples, max_n)     — valid neighbor mask
  dst_masks: (N_samples, max_n)     — valid destination mask

Usage:
  python collect_fv_data.py --min-n 8 --max-n 24 --graphs-per-nm 20 --out fv_data.npz
  python collect_fv_data.py --min-n 8 --max-n 24 --graphs-per-nm 20 --workers 62 --out fv_data.npz
"""

import argparse
import numpy as np
import sys
import time
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent))

from envs.gnm_env import (
    compute_gnm_features, compute_phi, algebraic_connectivity, NODE_FEAT_DIM,
    build_random_k_regular, build_random_tree_initial,
)
from utils.spectral import is_connected


def fv_expert_action(adj, degrees, n, u, v2, lambda2):
    """
    Compute the FV expert's optimal action at node u.

    Returns:
        (neighbor_action, destination, new_lambda2, new_v2, improved)
    """
    neighbors = np.where(adj[u] > 0)[0]
    non_neighbors = np.where((adj[u] == 0) & (np.arange(n) != u))[0]

    if len(neighbors) == 0 or len(non_neighbors) == 0:
        return n, 0, lambda2, v2, False

    nb_cut_scores = (v2[u] - v2[neighbors]) ** 2
    dst_cut_scores = (v2[u] - v2[non_neighbors]) ** 2

    nb_order = np.argsort(nb_cut_scores)
    dst_order = np.argsort(-dst_cut_scores)

    for nb_idx in nb_order[:3]:
        v = neighbors[nb_idx]
        for dst_idx in dst_order[:3]:
            w = non_neighbors[dst_idx]
            if dst_cut_scores[dst_idx] - nb_cut_scores[nb_idx] <= 0:
                break

            adj[u, v] = adj[v, u] = 0
            adj[u, w] = adj[w, u] = 1
            degrees[v] -= 1
            degrees[w] += 1

            if is_connected(adj):
                new_lambda2 = algebraic_connectivity(adj)
                if new_lambda2 > lambda2 + 1e-9:
                    L = np.diag(degrees) - adj
                    _, eigvecs = np.linalg.eigh(L)
                    new_v2 = eigvecs[:, 1]
                    if np.dot(new_v2, v2) < 0:
                        new_v2 = -new_v2
                    return int(v), int(w), new_lambda2, new_v2, True

            adj[u, v] = adj[v, u] = 1
            adj[u, w] = adj[w, u] = 0
            degrees[v] += 1
            degrees[w] -= 1

    return n, 0, lambda2, v2, False


def _collect_one_episode(args):
    """Worker: collect one episode. Returns packed numpy arrays."""
    n, m, seed, max_n_pad = args

    rng = np.random.RandomState(seed)

    if rng.random() < 0.5:
        adj = build_random_k_regular(n, m, rng)
    else:
        adj = build_random_tree_initial(n, m, rng)
    degrees = adj.sum(axis=1)

    L = np.diag(degrees) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    lambda2 = float(eigvals[1])
    v2 = eigvecs[:, 1]

    num_rounds = compute_phi(n, m)

    features_list = []
    nb_actions_list = []
    dst_actions_list = []
    nb_masks_list = []
    dst_masks_list = []

    for round_idx in range(num_rounds):
        round_frac = round_idx / max(num_rounds, 1)

        for u in range(n):
            feat = compute_gnm_features(adj, degrees, n, u, v2, round_frac)

            nb_mask = adj[u, :n].astype(bool)
            dst_mask = np.ones(n, dtype=bool)
            dst_mask[u] = False
            dst_mask[:n] &= ~nb_mask

            if not dst_mask.any():
                nb_action = n
                dst_action = 0
            else:
                nb_action, dst_action, new_lambda2, new_v2, improved = \
                    fv_expert_action(adj, degrees, n, u, v2, lambda2)
                if improved:
                    lambda2 = new_lambda2
                    v2 = new_v2

            feat_padded = np.zeros((max_n_pad, NODE_FEAT_DIM), dtype=np.float32)
            feat_padded[:n] = feat
            nb_mask_padded = np.zeros(max_n_pad, dtype=bool)
            nb_mask_padded[:n] = nb_mask
            dst_mask_padded = np.zeros(max_n_pad, dtype=bool)
            dst_mask_padded[:n] = dst_mask

            features_list.append(feat_padded)
            nb_actions_list.append(nb_action)
            dst_actions_list.append(dst_action)
            nb_masks_list.append(nb_mask_padded)
            dst_masks_list.append(dst_mask_padded)

    num_samples = len(features_list)
    if num_samples == 0:
        return None

    return (
        np.array(features_list, dtype=np.float32),
        np.array(nb_actions_list, dtype=np.int32),
        np.array(dst_actions_list, dtype=np.int32),
        np.array(nb_masks_list, dtype=bool),
        np.array(dst_masks_list, dtype=bool),
        np.full(num_samples, n, dtype=np.int32),
    )


def collect_data(min_n, max_n, graphs_per_nm, m_step, output_path, num_workers):
    """Collect FV expert data with multiprocessing."""
    max_n_pad = max_n

    # Build task list
    tasks = []
    for n in range(min_n, max_n + 1):
        min_m = n - 1
        max_m = n * (n - 1) // 2
        m_values = list(range(min_m, max_m + 1, m_step))
        if max_m not in m_values:
            m_values.append(max_m)
        for m in m_values:
            for g_idx in range(graphs_per_nm):
                seed = n * 100000 + m * 100 + g_idx
                tasks.append((n, m, seed, max_n_pad))

    total_tasks = len(tasks)
    print(f"Data collection: n=[{min_n},{max_n}], m_step={m_step}, "
          f"graphs/config={graphs_per_nm}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Workers: {num_workers}")
    print(f"  Max padded n: {max_n_pad}")
    print(f"  Feature dim: {NODE_FEAT_DIM}")
    sys.stdout.flush()

    t0 = time.time()

    all_feat = []
    all_nb = []
    all_dst = []
    all_nb_m = []
    all_dst_m = []
    all_n = []
    completed = 0

    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(_collect_one_episode, tasks, chunksize=16):
            completed += 1
            if result is not None:
                feat, nb_a, dst_a, nb_mask, dst_mask, n_vals = result
                all_feat.append(feat)
                all_nb.append(nb_a)
                all_dst.append(dst_a)
                all_nb_m.append(nb_mask)
                all_dst_m.append(dst_mask)
                all_n.append(n_vals)

            if completed % 500 == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                total_so_far = sum(len(x) for x in all_nb)
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{total_tasks} tasks ({100*completed/total_tasks:.0f}%), "
                      f"{total_so_far} samples, {elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s", flush=True)

    features = np.concatenate(all_feat, axis=0)
    nb_actions = np.concatenate(all_nb, axis=0)
    dst_actions = np.concatenate(all_dst, axis=0)
    nb_masks = np.concatenate(all_nb_m, axis=0)
    dst_masks = np.concatenate(all_dst_m, axis=0)
    n_vals = np.concatenate(all_n, axis=0)

    elapsed = time.time() - t0

    total_samples = len(nb_actions)
    total_rewires = int((nb_actions < n_vals).sum())
    total_keeps = total_samples - total_rewires

    np.savez_compressed(
        output_path,
        features=features,
        nb_actions=nb_actions,
        dst_actions=dst_actions,
        nb_masks=nb_masks,
        dst_masks=dst_masks,
        n_vals=n_vals,
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"FV Expert Data Collection Complete")
    print(f"{'='*60}")
    print(f"  n range:       [{min_n}, {max_n}]")
    print(f"  m step:        {m_step}")
    print(f"  graphs/config: {graphs_per_nm}")
    print(f"  tasks:         {total_tasks}")
    print(f"  samples:       {total_samples}")
    print(f"  rewires:       {total_rewires} ({100*total_rewires/total_samples:.1f}%)")
    print(f"  keeps:         {total_keeps} ({100*total_keeps/total_samples:.1f}%)")
    print(f"  time:          {elapsed:.1f}s ({total_tasks/elapsed:.1f} tasks/s)")
    print(f"  output:        {output_path} ({file_size_mb:.1f} MB)")
    print(f"  shape:         features={features.shape}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FV expert traces for SL pre-training")
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=24)
    parser.add_argument("--graphs-per-nm", type=int, default=20,
                        help="Number of random graphs per (n, m) config")
    parser.add_argument("--m-step", type=int, default=1,
                        help="Step size for m values (1=every m, 2=every other)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    parser.add_argument("--out", type=str, default="fv_expert_data.npz",
                        help="Output .npz file path")
    args = parser.parse_args()

    num_workers = args.workers or min(cpu_count(), 62)

    collect_data(
        min_n=args.min_n,
        max_n=args.max_n,
        graphs_per_nm=args.graphs_per_nm,
        m_step=args.m_step,
        output_path=args.out,
        num_workers=num_workers,
    )
