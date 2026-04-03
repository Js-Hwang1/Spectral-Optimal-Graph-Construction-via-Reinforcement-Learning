#!/usr/bin/env python3
"""
Sweep k for Rayleigh-Ritz subspace tracking.

Find the "knee" where adding more eigenvectors stops helping.
Tests k ∈ {2,3,4,5,6,7,8,10,12,16} at n=16,24,32.

Measures:
1. Greedy ADD quality (% of oracle Δλ₂)
2. Accumulated v₂ cosine similarity AUC over the full batch
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.spectral import exact_lambda2_np
from envs.gnm_env import build_random_tree_initial


def exact_eigenvectors(adj):
    L = np.diag(adj.sum(axis=1)) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs


def normalize_eigenvector(v):
    v = v / (np.linalg.norm(v) + 1e-15)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v


def rr_update(V, lams, u, v_node):
    """k-dim Rayleigh-Ritz update. O(Nk²)."""
    delta = V[u, :] - V[v_node, :]
    L_sub = np.diag(lams) + np.outer(delta, delta)
    new_lams, R = np.linalg.eigh(L_sub)
    V_new = V @ R
    return V_new, new_lams


def sweep_greedy_add(n, n_trials, k_values):
    """Greedy ADD quality for different k values."""
    rng = np.random.RandomState(42)
    m = n * (n - 1) // 4  # ~50% density
    n_edges = n  # batch size = n edges (like delta=2 × n/2)

    dl2 = {k: [] for k in k_values}
    dl2['stale'] = []
    dl2['oracle'] = []

    for t in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        l2_init = exact_lambda2_np(adj)
        eigvals, eigvecs = exact_eigenvectors(adj)
        v2_orig = eigvecs[:, 1].copy()
        n_eig = len(eigvals)

        # Stale
        adj_s = adj.copy()
        ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_s[i,j]==0]
        sc = {e: (v2_orig[e[0]]-v2_orig[e[1]])**2 for e in ne}
        top = sorted(sc.keys(), key=lambda e: sc[e], reverse=True)
        for i, j in top[:n_edges]:
            adj_s[i,j] = 1; adj_s[j,i] = 1
        dl2['stale'].append(exact_lambda2_np(adj_s) - l2_init)

        # RR for each k
        for k in k_values:
            adj_r = adj.copy()
            kk = min(k, n_eig - 1)
            V = eigvecs[:, 1:1+kk].copy()
            lams = eigvals[1:1+kk].copy()
            for _ in range(n_edges):
                ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_r[i,j]==0]
                if not ne: break
                v2c = V[:, 0]
                bs = -1; be = None
                for i, j in ne:
                    g = (v2c[i]-v2c[j])**2
                    if g > bs: bs = g; be = (i,j)
                u, v = be
                V, lams = rr_update(V, lams, u, v)
                adj_r[u,v] = 1; adj_r[v,u] = 1
            dl2[k].append(exact_lambda2_np(adj_r) - l2_init)

        # Oracle
        adj_x = adj.copy()
        for _ in range(n_edges):
            ev, evec = exact_eigenvectors(adj_x)
            v2k = evec[:, 1]
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_x[i,j]==0]
            if not ne: break
            bg = -1; be = None
            for i, j in ne:
                g = (v2k[i]-v2k[j])**2
                if g > bg: bg = g; be = (i,j)
            u, v = be
            adj_x[u,v] = 1; adj_x[v,u] = 1
        dl2['oracle'].append(exact_lambda2_np(adj_x) - l2_init)

    return dl2


def sweep_cosine_auc(n, n_trials, k_values):
    """Accumulated v₂ cosine similarity per edge step (greedy edges)."""
    rng = np.random.RandomState(42)
    m = n * (n - 1) // 4
    n_edges = n

    # cos_sim[k][step] = list of cosine sims
    cos_data = {k: {s: [] for s in range(1, n_edges+1)} for k in k_values}

    for t in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)
        v2_orig = eigvecs[:, 1].copy()
        n_eig = len(eigvals)

        # Initialize all trackers
        states = {}
        for k in k_values:
            kk = min(k, n_eig - 1)
            states[k] = {
                'V': eigvecs[:, 1:1+kk].copy(),
                'lams': eigvals[1:1+kk].copy(),
            }

        adj_work = adj.copy()
        for s in range(1, n_edges + 1):
            # Greedy edge using stale v₂ (hardest test)
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_work[i,j]==0]
            if not ne: break
            bg = -1; be = None
            for i, j in ne:
                g = (v2_orig[i]-v2_orig[j])**2
                if g > bg: bg = g; be = (i,j)
            u, v = be

            for k in k_values:
                st = states[k]
                st['V'], st['lams'] = rr_update(st['V'], st['lams'], u, v)

            adj_work[u,v] = 1; adj_work[v,u] = 1

            eigvals_true, eigvecs_true = exact_eigenvectors(adj_work)
            v2_true = normalize_eigenvector(eigvecs_true[:, 1])

            for k in k_values:
                v2_approx = normalize_eigenvector(states[k]['V'][:, 0])
                cos_data[k][s].append(abs(np.dot(v2_approx, v2_true)))

    return cos_data


if __name__ == "__main__":
    k_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]

    # =====================================================================
    # Part 1: Greedy ADD quality sweep
    # =====================================================================
    for n, n_trials in [(16, 80), (24, 60), (32, 40)]:
        m = n * (n-1) // 4
        print(f"\n{'='*70}")
        print(f"GREEDY ADD QUALITY: n={n}, m={m}, batch={n} edges, {n_trials} trials")
        print(f"{'='*70}")

        dl2 = sweep_greedy_add(n, n_trials, k_values)
        mo = np.mean(dl2['oracle'])
        ms = np.mean(dl2['stale'])

        print(f"  {'k':>4s} | {'avg Δλ₂':>10s} | {'% oracle':>9s} | {'gain over stale':>16s}")
        print(f"  {'-'*4} | {'-'*10} | {'-'*9} | {'-'*16}")
        print(f"  {'stl':>4s} | {ms:+10.3f} | {ms/mo*100:8.1f}% | {'baseline':>16s}")
        prev_pct = ms/mo*100
        for k in k_values:
            mk = np.mean(dl2[k])
            pct = mk/mo*100 if mo > 0 else 0
            gain = pct - prev_pct
            print(f"  {k:4d} | {mk:+10.3f} | {pct:8.1f}% | {'+' if gain>=0 else ''}{gain:.1f}pp")
            prev_pct = pct
        print(f"  {'orc':>4s} | {mo:+10.3f} | {'100.0':>8s}% |")

    # =====================================================================
    # Part 2: Cosine AUC at n=32
    # =====================================================================
    n = 32
    n_trials = 40
    print(f"\n{'='*70}")
    print(f"COSINE AUC: n={n}, greedy edges, {n_trials} trials")
    print(f"{'='*70}")

    cos_data = sweep_cosine_auc(n, n_trials, k_values)

    # Print AUC and mean cosine at key checkpoints
    print(f"\n  Mean cosine sim at edge steps:")
    header = f"  {'step':>5s}"
    for k in k_values:
        header += f" {'k='+str(k):>6s}"
    print(header)
    print(f"  {'-'*5}" + (" " + "-"*6) * len(k_values))

    checkpoints = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    for s in checkpoints:
        if s > n: break
        line = f"  {s:5d}"
        for k in k_values:
            vals = cos_data[k].get(s, [])
            if vals:
                line += f" {np.mean(vals):6.3f}"
            else:
                line += f" {'N/A':>6s}"
        print(line)

    # AUC (average cosine over all steps)
    print(f"\n  AUC (mean cosine across all {n} steps):")
    header = f"  "
    for k in k_values:
        header += f" {'k='+str(k):>6s}"
    print(header)
    line = f"  "
    for k in k_values:
        all_cos = []
        for s in range(1, n+1):
            all_cos.extend(cos_data[k].get(s, []))
        if all_cos:
            line += f" {np.mean(all_cos):6.3f}"
        else:
            line += f" {'N/A':>6s}"
    print(line)

    # Marginal gain
    print(f"\n  Marginal gain in AUC from k to k+1:")
    aucs = []
    for k in k_values:
        all_cos = []
        for s in range(1, n+1):
            all_cos.extend(cos_data[k].get(s, []))
        aucs.append(np.mean(all_cos) if all_cos else 0)

    for i in range(1, len(k_values)):
        gain = aucs[i] - aucs[i-1]
        print(f"    k={k_values[i-1]}→{k_values[i]}: {gain:+.4f}")
