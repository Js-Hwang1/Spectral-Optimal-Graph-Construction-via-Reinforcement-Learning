#!/usr/bin/env python3
"""
PoC: k-dimensional Rayleigh-Ritz Subspace Tracking

Extends the 2D Rayleigh-Ritz from rayleigh_ritz_poc.py to k dimensions.

When adding edge (u,v), the Laplacian update is rank-1: L_new = L + zz^T.
Track V = [v₂, ..., v_{k+1}] ∈ R^{N×k} and eigenvalues λ₂..λ_{k+1}.

Update:
  δ_m = v_m[u] - v_m[v]  for m in {2..k+1}     # O(k) — the gap vector
  L_sub = diag(λ₂..λ_{k+1}) + δδ^T              # O(k²) — rank-1 update to k×k
  Diagonalize L_sub → (eigenvalues, R)            # O(k³) — constant for k≤8
  V_new = V @ R                                   # O(Nk²) — rotate full vectors

Total per edge: O(Nk²). Over n·delta edges: O(N·n·delta·k²) = O(N²·k²) since delta is const.
For k=5: O(25N²) — well within O(N²) budget.

Tests k=2,3,4,5 side by side on greedy ADD quality and accumulated accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.spectral import exact_lambda2_np
from envs.gnm_env import build_random_tree_initial


def exact_eigenvectors(adj):
    """Compute exact L eigenvectors and eigenvalues."""
    L = np.diag(adj.sum(axis=1)) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs


def normalize_eigenvector(v):
    """Normalize and fix sign ambiguity."""
    v = v / (np.linalg.norm(v) + 1e-15)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v


def rr_update(V, lams, u, v_node):
    """
    k-dimensional Rayleigh-Ritz update after adding edge (u, v_node).

    Args:
        V: (N, k) matrix of tracked eigenvectors [v₂..v_{k+1}]
        lams: (k,) array of tracked eigenvalues [λ₂..λ_{k+1}]
        u, v_node: edge endpoints

    Returns:
        V_new: (N, k) updated eigenvectors
        lams_new: (k,) updated eigenvalues

    Complexity: O(Nk²) for rotation, O(k³) for eigensolver (constant for small k).
    """
    k = V.shape[1]

    # Gap vector: δ_m = V[u,m] - V[v_node,m] for each tracked eigenvector
    delta = V[u, :] - V[v_node, :]  # (k,)

    # Build k×k projected Laplacian: diag(lams) + δδ^T
    L_sub = np.diag(lams) + np.outer(delta, delta)  # (k, k)

    # Diagonalize — O(k³), constant for k≤8
    new_lams, R = np.linalg.eigh(L_sub)  # new_lams sorted ascending, R columns are eigvecs

    # Rotate full N-dim vectors: V_new = V @ R  — O(Nk²)
    V_new = V @ R

    return V_new, new_lams


# ============================================================================
# Test 1: Accumulated accuracy for k=2,3,4,5
# ============================================================================
def test_accumulated(n=8, m=10, n_edges=8, n_trials=200):
    """Track v₂ cosine similarity over sequential random edges for different k."""
    rng = np.random.RandomState(42)
    k_values = [2, 3, 4, 5]

    results = {k: {e: [] for e in range(1, n_edges+1)} for k in k_values}
    lam_results = {k: {e: [] for e in range(1, n_edges+1)} for k in k_values}

    for _ in range(n_trials):
        adj_orig = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj_orig)
        n_eig = len(eigvals)

        # Initialize state for each k
        states = {}
        for k in k_values:
            kk = min(k, n_eig - 1)  # can't track more eigenvectors than exist
            states[k] = {
                'V': eigvecs[:, 1:1+kk].copy(),  # (N, kk)
                'lams': eigvals[1:1+kk].copy(),   # (kk,)
            }

        adj = adj_orig.copy()
        for e_idx in range(1, n_edges + 1):
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
            if not non_edges:
                break
            u, v = non_edges[rng.randint(len(non_edges))]

            # Update each k-tracker
            for k in k_values:
                st = states[k]
                st['V'], st['lams'] = rr_update(st['V'], st['lams'], u, v)

            # Apply edge
            adj[u, v] = 1; adj[v, u] = 1

            # Ground truth
            eigvals_true, eigvecs_true = exact_eigenvectors(adj)
            v2_true = normalize_eigenvector(eigvecs_true[:, 1])
            lam2_true = eigvals_true[1]

            for k in k_values:
                v2_approx = normalize_eigenvector(states[k]['V'][:, 0])
                cos_sim = abs(np.dot(v2_approx, v2_true))
                results[k][e_idx].append(cos_sim)
                lam_err = abs(states[k]['lams'][0] - lam2_true)
                lam_results[k][e_idx].append(lam_err)

    print("=" * 70)
    print(f"TEST 1: Accumulated v₂ accuracy (n={n}, m={m}, random edges)")
    print("=" * 70)
    header = f"  {'edges':>5s}"
    for k in k_values:
        header += f" | {'k='+str(k)+' cos':>8s} {'>0.9':>6s} {'λ₂err':>7s}"
    print(header)
    print(f"  {'-'*5}" + (" | " + "-"*8 + " " + "-"*6 + " " + "-"*7) * len(k_values))

    for e_idx in range(1, n_edges + 1):
        line = f"  {e_idx:5d}"
        for k in k_values:
            vals = results[k][e_idx]
            lvals = lam_results[k][e_idx]
            if vals:
                cos_m = np.mean(vals)
                pct90 = sum(1 for c in vals if c > 0.90) / len(vals) * 100
                lam_m = np.mean(lvals)
                line += f" | {cos_m:8.4f} {pct90:5.1f}% {lam_m:7.4f}"
            else:
                line += f" | {'N/A':>8s} {'N/A':>6s} {'N/A':>7s}"
        print(line)
    print()


# ============================================================================
# Test 2: Greedy ADD quality for k=2,3,4,5
# ============================================================================
def test_greedy_add(n=8, m=14, n_edges=8, n_trials=300):
    """Compare greedy ADD quality using RR with different k values."""
    rng = np.random.RandomState(42)
    k_values = [2, 3, 4, 5]

    dl2 = {k: [] for k in k_values}
    dl2['stale'] = []
    dl2['oracle'] = []
    dl2['random'] = []

    for _ in range(n_trials):
        adj_orig = build_random_tree_initial(n, m, rng)
        l2_init = exact_lambda2_np(adj_orig)
        eigvals, eigvecs = exact_eigenvectors(adj_orig)
        v2_orig = eigvecs[:, 1].copy()
        n_eig = len(eigvals)

        # --- Stale greedy ---
        adj = adj_orig.copy()
        ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
        sc = {e: (v2_orig[e[0]]-v2_orig[e[1]])**2 for e in ne}
        top = sorted(sc.keys(), key=lambda e: sc[e], reverse=True)
        for i, j in top[:n_edges]:
            adj[i,j] = 1; adj[j,i] = 1
        dl2['stale'].append(exact_lambda2_np(adj) - l2_init)

        # --- RR greedy for each k ---
        for k in k_values:
            adj = adj_orig.copy()
            kk = min(k, n_eig - 1)
            V = eigvecs[:, 1:1+kk].copy()
            lams = eigvals[1:1+kk].copy()

            for _ in range(n_edges):
                ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
                if not ne: break
                # Score using tracked v₂ (first column of V)
                v2_curr = V[:, 0]
                best_s = -1; best_e = None
                for i, j in ne:
                    g = (v2_curr[i] - v2_curr[j])**2
                    if g > best_s: best_s = g; best_e = (i,j)
                u, v = best_e
                V, lams = rr_update(V, lams, u, v)
                adj[u,v] = 1; adj[v,u] = 1
            dl2[k].append(exact_lambda2_np(adj) - l2_init)

        # --- Oracle greedy ---
        adj = adj_orig.copy()
        for _ in range(n_edges):
            ev, evec = exact_eigenvectors(adj)
            v2k = evec[:, 1]
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            if not ne: break
            bg = -1; be = None
            for i, j in ne:
                g = (v2k[i]-v2k[j])**2
                if g > bg: bg = g; be = (i,j)
            u, v = be
            adj[u,v] = 1; adj[v,u] = 1
        dl2['oracle'].append(exact_lambda2_np(adj) - l2_init)

        # --- Random ---
        adj = adj_orig.copy()
        ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
        rng.shuffle(ne)
        for i, j in ne[:n_edges]:
            adj[i,j] = 1; adj[j,i] = 1
        dl2['random'].append(exact_lambda2_np(adj) - l2_init)

    print("=" * 70)
    print(f"TEST 2: Greedy ADD quality — {n_edges} edges (n={n}, m={m})")
    print("=" * 70)
    oracle_mean = np.mean(dl2['oracle'])
    print(f"  {'Method':<25s} | {'avg Δλ₂':>10s} | {'vs oracle':>10s}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10}")
    for label, key in [
        ("Random", 'random'),
        ("Stale greedy", 'stale'),
        ("RR k=2", 2),
        ("RR k=3", 3),
        ("RR k=4", 4),
        ("RR k=5", 5),
        ("Oracle greedy", 'oracle'),
    ]:
        m_val = np.mean(dl2[key])
        pct = m_val / oracle_mean * 100 if oracle_mean > 0 else 0
        print(f"  {label:<25s} | {m_val:+10.4f} | {pct:9.1f}%")
    print()


# ============================================================================
# Test 3: Scaling — k=2 vs k=5 at different n
# ============================================================================
def test_scaling(n_values=[8, 12, 16, 24, 32], n_trials=100):
    """Compare RR k=2 vs k=5 at different graph sizes."""
    rng = np.random.RandomState(42)
    k_values = [2, 5]

    print("=" * 70)
    print("TEST 3: Scaling — RR k=2 vs k=5 vs stale vs oracle")
    print("=" * 70)
    print(f"  {'n':>4s} {'m':>5s} {'#add':>5s} | {'Stale':>7s} {'RR k=2':>7s} {'RR k=5':>7s} {'Oracle':>7s} | {'S/O':>5s} {'k2/O':>5s} {'k5/O':>5s}")
    print(f"  {'-'*4} {'-'*5} {'-'*5} | {'-'*7} {'-'*7} {'-'*7} {'-'*7} | {'-'*5} {'-'*5} {'-'*5}")

    for n in n_values:
        m = n * (n - 1) // 4
        n_edges = n

        dl2_s = []; dl2_k = {k: [] for k in k_values}; dl2_o = []

        for _ in range(n_trials):
            adj = build_random_tree_initial(n, m, rng)
            l2_init = exact_lambda2_np(adj)
            eigvals, eigvecs = exact_eigenvectors(adj)
            v2_o = eigvecs[:, 1].copy()
            n_eig = len(eigvals)

            # Stale
            adj_s = adj.copy()
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            sc = {e: (v2_o[e[0]]-v2_o[e[1]])**2 for e in ne}
            top = sorted(sc.keys(), key=lambda e: sc[e], reverse=True)
            for i, j in top[:n_edges]:
                adj_s[i,j] = 1; adj_s[j,i] = 1
            dl2_s.append(exact_lambda2_np(adj_s) - l2_init)

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
                dl2_k[k].append(exact_lambda2_np(adj_r) - l2_init)

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
            dl2_o.append(exact_lambda2_np(adj_x) - l2_init)

        ms = np.mean(dl2_s)
        m2 = np.mean(dl2_k[2])
        m5 = np.mean(dl2_k[5])
        mo = np.mean(dl2_o)
        sp = ms/mo*100 if mo>0 else 0
        k2p = m2/mo*100 if mo>0 else 0
        k5p = m5/mo*100 if mo>0 else 0
        print(f"  {n:4d} {m:5d} {n_edges:5d} | {ms:+7.3f} {m2:+7.3f} {m5:+7.3f} {mo:+7.3f} | {sp:4.0f}% {k2p:4.0f}% {k5p:4.0f}%")
    print()


# ============================================================================
# Test 4: Accumulated v₂ accuracy — greedy edges (worst case)
# ============================================================================
def test_accumulated_greedy(n=8, m=10, n_edges=8, n_trials=200):
    """
    Track v₂ accuracy when edges are chosen GREEDILY (worst case for staleness).
    Greedy edges maximally change the Fiedler vector — hardest test for tracking.
    """
    rng = np.random.RandomState(42)
    k_values = [2, 3, 4, 5]

    results = {k: {e: [] for e in range(1, n_edges+1)} for k in k_values}

    for _ in range(n_trials):
        adj_orig = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj_orig)
        v2_orig = eigvecs[:, 1].copy()
        n_eig = len(eigvals)

        # Initialize states
        states = {}
        for k in k_values:
            kk = min(k, n_eig - 1)
            states[k] = {
                'V': eigvecs[:, 1:1+kk].copy(),
                'lams': eigvals[1:1+kk].copy(),
                'adj': adj_orig.copy(),
            }

        for e_idx in range(1, n_edges + 1):
            # Pick edge greedily using STALE v₂ (worst case — all methods see same edge)
            adj_ref = states[k_values[0]]['adj']
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_ref[i,j]==0]
            if not ne: break
            bg = -1; be = None
            for i, j in ne:
                g = (v2_orig[i]-v2_orig[j])**2
                if g > bg: bg = g; be = (i,j)
            u, v = be

            # Update all trackers with the same edge
            for k in k_values:
                st = states[k]
                st['V'], st['lams'] = rr_update(st['V'], st['lams'], u, v)
                st['adj'][u, v] = 1; st['adj'][v, u] = 1

            # Ground truth
            eigvals_true, eigvecs_true = exact_eigenvectors(states[k_values[0]]['adj'])
            v2_true = normalize_eigenvector(eigvecs_true[:, 1])

            for k in k_values:
                v2_approx = normalize_eigenvector(states[k]['V'][:, 0])
                cos_sim = abs(np.dot(v2_approx, v2_true))
                results[k][e_idx].append(cos_sim)

    print("=" * 70)
    print(f"TEST 4: Accumulated v₂ accuracy — GREEDY edges (n={n}, m={m})")
    print("=" * 70)
    header = f"  {'edges':>5s}"
    for k in k_values:
        header += f" | {'k='+str(k):>6s} {'>0.9':>6s}"
    print(header)
    print(f"  {'-'*5}" + (" | " + "-"*6 + " " + "-"*6) * len(k_values))

    for e_idx in range(1, n_edges + 1):
        line = f"  {e_idx:5d}"
        for k in k_values:
            vals = results[k][e_idx]
            if vals:
                cos_m = np.mean(vals)
                pct90 = sum(1 for c in vals if c > 0.90) / len(vals) * 100
                line += f" | {cos_m:6.4f} {pct90:5.1f}%"
        print(line)
    print()


# ============================================================================
# Test 5: Scaling with greedy — the real bottom line
# ============================================================================
def test_scaling_greedy(n_values=[8, 12, 16, 24, 32, 48], n_trials=80):
    """
    Greedy ADD using RR-tracked v₂ at different n, for k=2,3,4,5.
    Each edge is chosen greedily using the TRACKED v₂ (not stale).
    """
    rng = np.random.RandomState(42)
    k_values = [2, 3, 4, 5]

    print("=" * 70)
    print("TEST 5: Scaling — Greedy ADD with RR-tracked v₂")
    print("=" * 70)
    header = f"  {'n':>4s} {'m':>5s} {'#add':>5s} | {'Stale':>6s}"
    for k in k_values:
        header += f" {'k='+str(k):>6s}"
    header += f" {'Oracle':>7s} |"
    for k in k_values:
        header += f" {'k'+str(k)+'/O':>5s}"
    print(header)
    divider = f"  {'-'*4} {'-'*5} {'-'*5} | {'-'*6}"
    for _ in k_values:
        divider += f" {'-'*6}"
    divider += f" {'-'*7} |"
    for _ in k_values:
        divider += f" {'-'*5}"
    print(divider)

    for n in n_values:
        m = n * (n - 1) // 4
        n_edges = n
        n_eig_max = n  # at most n eigenvectors

        dl2_s = []; dl2_k = {k: [] for k in k_values}; dl2_o = []

        for _ in range(n_trials):
            adj = build_random_tree_initial(n, m, rng)
            l2_init = exact_lambda2_np(adj)
            eigvals, eigvecs = exact_eigenvectors(adj)
            v2_orig = eigvecs[:, 1].copy()

            # Stale
            adj_s = adj.copy()
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_s[i,j]==0]
            sc = {e: (v2_orig[e[0]]-v2_orig[e[1]])**2 for e in ne}
            top = sorted(sc.keys(), key=lambda e: sc[e], reverse=True)
            for i, j in top[:n_edges]:
                adj_s[i,j] = 1; adj_s[j,i] = 1
            dl2_s.append(exact_lambda2_np(adj_s) - l2_init)

            # RR for each k — greedy using TRACKED v₂
            for k in k_values:
                adj_r = adj.copy()
                kk = min(k, n_eig_max - 1)
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
                dl2_k[k].append(exact_lambda2_np(adj_r) - l2_init)

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
            dl2_o.append(exact_lambda2_np(adj_x) - l2_init)

        ms = np.mean(dl2_s)
        mo = np.mean(dl2_o)
        line = f"  {n:4d} {m:5d} {n_edges:5d} | {ms:+6.2f}"
        pcts = []
        for k in k_values:
            mk = np.mean(dl2_k[k])
            line += f" {mk:+6.2f}"
            pcts.append(mk/mo*100 if mo>0 else 0)
        line += f" {mo:+7.2f} |"
        for p in pcts:
            line += f" {p:4.0f}%"
        print(line)
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("  k-dimensional Rayleigh-Ritz Subspace Tracking PoC")
    print("=" * 70)
    print()

    test_accumulated(n=8, m=10, n_edges=8)
    test_greedy_add(n=8, m=14, n_edges=8)
    test_greedy_add(n=8, m=10, n_edges=8)
    test_accumulated_greedy(n=8, m=10, n_edges=8)
    test_scaling(n_trials=80)
    test_scaling_greedy(n_values=[8, 12, 16, 24, 32, 48], n_trials=60)
