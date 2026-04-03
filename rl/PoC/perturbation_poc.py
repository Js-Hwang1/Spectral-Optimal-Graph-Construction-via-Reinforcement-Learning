#!/usr/bin/env python3
"""
PoC: Rank-1 Perturbation Correction for Autoregressive Edge Placement

Tests whether first-order perturbation theory can cheaply track the Fiedler
vector through sequential edge additions, avoiding stale-v₂ degradation.

Key formula:
  After adding edge (u,v):
    α = (v₂[u]-v₂[v]) · (v₃[u]-v₃[v]) / (λ₂ - λ₃)
    v₂_new ≈ v₂ + α · v₃           (O(N) update)
    λ₂_new ≈ λ₂ + (v₂[u]-v₂[v])²  (O(1) update)

Tests:
  1. Accuracy: how well does the perturbation-corrected v₂ match the true v₂?
  2. Scoring: does corrected_gap² rank edges better than stale_gap²?
  3. Greedy quality: does greedy-with-correction beat greedy-with-stale?
  4. Accumulation: does the approximation hold over many sequential edges?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.spectral import exact_lambda2_np, lanczos_fiedler_ext
from envs.gnm_env import build_random_tree_initial


def exact_eigenvectors(adj):
    """Compute exact L eigenvectors and eigenvalues."""
    L = np.diag(adj.sum(axis=1)) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs


def perturbation_alpha(v2, v3, lam2, lam3, u, v):
    """Compute rotation parameter α for adding edge (u,v). O(1)."""
    gap2 = v2[u] - v2[v]
    gap3 = v3[u] - v3[v]
    denom = lam2 - lam3
    if abs(denom) < 1e-12:
        return 0.0
    return gap2 * gap3 / denom


def normalize_eigenvector(v):
    """Normalize and fix sign ambiguity (make largest component positive)."""
    v = v / (np.linalg.norm(v) + 1e-15)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v


# ============================================================================
# Test 1: Single edge — perturbation accuracy
# ============================================================================
def test_single_edge_accuracy(n=8, m=14, n_trials=500):
    """How well does v₂ + α·v₃ approximate the true v₂ after one edge add?"""
    rng = np.random.RandomState(42)

    cosine_sims = []
    lam2_errors = []

    for _ in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)
        v2 = eigvecs[:, 1]
        v3 = eigvecs[:, 2]
        lam2 = eigvals[1]
        lam3 = eigvals[2]

        # Pick a random non-edge
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        if not non_edges:
            continue
        u, v = non_edges[rng.randint(len(non_edges))]

        # Perturbation prediction
        alpha = perturbation_alpha(v2, v3, lam2, lam3, u, v)
        v2_pred = v2 + alpha * v3
        v2_pred = normalize_eigenvector(v2_pred)
        lam2_pred = lam2 + (v2[u] - v2[v]) ** 2

        # True values after adding edge
        adj[u, v] = 1; adj[v, u] = 1
        eigvals_new, eigvecs_new = exact_eigenvectors(adj)
        v2_true = normalize_eigenvector(eigvecs_new[:, 1])
        lam2_true = eigvals_new[1]

        cosine_sim = abs(np.dot(v2_pred, v2_true))
        cosine_sims.append(cosine_sim)
        lam2_errors.append(abs(lam2_pred - lam2_true))

    print("=" * 70)
    print(f"TEST 1: Single-edge perturbation accuracy (n={n}, m={m})")
    print("=" * 70)
    print(f"  v₂ cosine similarity: {np.mean(cosine_sims):.4f} ± {np.std(cosine_sims):.4f}")
    print(f"  v₂ cosine sim > 0.95: {sum(1 for c in cosine_sims if c > 0.95)/len(cosine_sims)*100:.1f}%")
    print(f"  v₂ cosine sim > 0.99: {sum(1 for c in cosine_sims if c > 0.99)/len(cosine_sims)*100:.1f}%")
    print(f"  λ₂ abs error:         {np.mean(lam2_errors):.4f} ± {np.std(lam2_errors):.4f}")
    print()
    return np.mean(cosine_sims)


# ============================================================================
# Test 2: Accumulated edges — does approximation hold?
# ============================================================================
def test_accumulated_accuracy(n=8, m=10, n_edges=8, n_trials=200):
    """How well does accumulated perturbation track v₂ over many edges?"""
    rng = np.random.RandomState(42)

    results_by_k = {k: [] for k in range(1, n_edges + 1)}

    for _ in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)
        v2_orig = eigvecs[:, 1].copy()
        v3_orig = eigvecs[:, 2].copy()
        lam2 = eigvals[1]
        lam3 = eigvals[2]

        # Track accumulated perturbation
        alpha_total = 0.0

        for k in range(1, n_edges + 1):
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
            if not non_edges:
                break
            u, v = non_edges[rng.randint(len(non_edges))]

            # Accumulate α using ORIGINAL v₂, v₃
            alpha_k = perturbation_alpha(v2_orig, v3_orig, lam2, lam3, u, v)
            alpha_total += alpha_k

            # Update λ₂ estimate
            lam2 += (v2_orig[u] - v2_orig[v]) ** 2
            lam3 += (v3_orig[u] - v3_orig[v]) ** 2

            # Apply edge
            adj[u, v] = 1; adj[v, u] = 1

            # Perturbation prediction
            v2_pred = v2_orig + alpha_total * v3_orig
            v2_pred = normalize_eigenvector(v2_pred)

            # True v₂
            eigvals_new, eigvecs_new = exact_eigenvectors(adj)
            v2_true = normalize_eigenvector(eigvecs_new[:, 1])

            cosine_sim = abs(np.dot(v2_pred, v2_true))
            results_by_k[k].append(cosine_sim)

    print("=" * 70)
    print(f"TEST 2: Accumulated perturbation (n={n}, m={m}, up to {n_edges} edges)")
    print("=" * 70)
    print(f"  {'edges':>5s} | {'cos_sim':>8s} | {'> 0.90':>7s} | {'> 0.80':>7s}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*7} | {'-'*7}")
    for k in range(1, n_edges + 1):
        vals = results_by_k[k]
        if vals:
            pct90 = sum(1 for c in vals if c > 0.90) / len(vals) * 100
            pct80 = sum(1 for c in vals if c > 0.80) / len(vals) * 100
            print(f"  {k:5d} | {np.mean(vals):8.4f} | {pct90:6.1f}% | {pct80:6.1f}%")
    print()


# ============================================================================
# Test 3: Scoring quality — does corrected gap rank edges better?
# ============================================================================
def test_scoring_quality(n=8, m=14, n_edges_placed=4, n_trials=300):
    """
    After placing n_edges_placed edges, compare ranking quality of:
    a) Stale v₂ gap (no correction)
    b) Perturbation-corrected v₂ gap
    c) True v₂ gap (oracle)

    Metric: Kendall tau correlation with oracle ranking of remaining non-edges.
    """
    from scipy.stats import kendalltau
    rng = np.random.RandomState(42)

    tau_stale = []
    tau_corrected = []

    for _ in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)
        v2_orig = eigvecs[:, 1].copy()
        v3_orig = eigvecs[:, 2].copy()
        lam2 = eigvals[1]
        lam3 = eigvals[2]

        alpha_total = 0.0

        # Place n_edges_placed edges greedily (using stale v₂)
        for _ in range(n_edges_placed):
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
            if not non_edges:
                break
            # Greedy: pick edge with max stale v₂ gap
            best_gap = -1
            best_e = None
            for i, j in non_edges:
                gap = (v2_orig[i] - v2_orig[j]) ** 2
                if gap > best_gap:
                    best_gap = gap
                    best_e = (i, j)

            u, v = best_e
            alpha_total += perturbation_alpha(v2_orig, v3_orig, lam2, lam3, u, v)
            lam2 += (v2_orig[u] - v2_orig[v]) ** 2
            adj[u, v] = 1; adj[v, u] = 1

        # Now score remaining non-edges three ways
        remaining = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        if len(remaining) < 3:
            continue

        # Stale scores
        stale_scores = [n * (v2_orig[i] - v2_orig[j]) ** 2 for i, j in remaining]

        # Corrected scores
        corrected_scores = []
        for i, j in remaining:
            gap_corrected = (v2_orig[i] - v2_orig[j]) + alpha_total * (v3_orig[i] - v3_orig[j])
            corrected_scores.append(n * gap_corrected ** 2)

        # Oracle scores (true v₂ of current graph)
        eigvals_now, eigvecs_now = exact_eigenvectors(adj)
        v2_true = eigvecs_now[:, 1]
        oracle_scores = [n * (v2_true[i] - v2_true[j]) ** 2 for i, j in remaining]

        # Kendall tau with oracle
        t_stale, _ = kendalltau(stale_scores, oracle_scores)
        t_corrected, _ = kendalltau(corrected_scores, oracle_scores)

        if not np.isnan(t_stale) and not np.isnan(t_corrected):
            tau_stale.append(t_stale)
            tau_corrected.append(t_corrected)

    print("=" * 70)
    print(f"TEST 3: Scoring quality after {n_edges_placed} edges (n={n}, m={m})")
    print("=" * 70)
    print(f"  Kendall τ with oracle (higher = better ranking):")
    print(f"    Stale v₂:     {np.mean(tau_stale):+.4f} ± {np.std(tau_stale):.4f}")
    print(f"    Corrected v₂: {np.mean(tau_corrected):+.4f} ± {np.std(tau_corrected):.4f}")
    improvement = np.mean(tau_corrected) - np.mean(tau_stale)
    print(f"    Improvement:  {improvement:+.4f}")
    print()
    return np.mean(tau_stale), np.mean(tau_corrected)


# ============================================================================
# Test 4: Greedy ADD quality — the bottom line
# ============================================================================
def test_greedy_add_quality(n=8, m=14, n_edges=8, n_trials=300):
    """
    Compare total Δλ₂ from adding n_edges using:
    a) Stale greedy: all edges scored with original v₂
    b) Corrected greedy: perturbation-corrected v₂ gap
    c) Oracle greedy: recompute exact v₂ after each edge
    d) Random: random non-edge selection
    """
    rng = np.random.RandomState(42)

    dl2_stale = []
    dl2_corrected = []
    dl2_oracle = []
    dl2_random = []

    for _ in range(n_trials):
        adj_orig = build_random_tree_initial(n, m, rng)
        l2_init = exact_lambda2_np(adj_orig)

        eigvals, eigvecs = exact_eigenvectors(adj_orig)
        v2_orig = eigvecs[:, 1].copy()
        v3_orig = eigvecs[:, 2].copy()
        lam2_orig = eigvals[1]
        lam3_orig = eigvals[2]

        # --- Stale greedy ---
        adj = adj_orig.copy()
        # Score all non-edges once, add top-k
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        stale_scores = {(i,j): (v2_orig[i]-v2_orig[j])**2 for i,j in non_edges}
        sorted_ne = sorted(stale_scores.keys(), key=lambda e: stale_scores[e], reverse=True)
        for i, j in sorted_ne[:n_edges]:
            adj[i,j] = 1; adj[j,i] = 1
        dl2_stale.append(exact_lambda2_np(adj) - l2_init)

        # --- Corrected greedy ---
        adj = adj_orig.copy()
        alpha_total = 0.0
        lam2 = lam2_orig
        lam3 = lam3_orig
        for _ in range(n_edges):
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
            if not non_edges:
                break
            # Score with corrected v₂ gap
            best_score = -1
            best_e = None
            for i, j in non_edges:
                gap = (v2_orig[i]-v2_orig[j]) + alpha_total * (v3_orig[i]-v3_orig[j])
                score = gap ** 2
                if score > best_score:
                    best_score = score
                    best_e = (i, j)
            u, v = best_e
            alpha_total += perturbation_alpha(v2_orig, v3_orig, lam2, lam3, u, v)
            lam2 += (v2_orig[u] - v2_orig[v]) ** 2
            lam3 += (v3_orig[u] - v3_orig[v]) ** 2
            adj[u, v] = 1; adj[v, u] = 1
        dl2_corrected.append(exact_lambda2_np(adj) - l2_init)

        # --- Oracle greedy ---
        adj = adj_orig.copy()
        for _ in range(n_edges):
            eigvals_k, eigvecs_k = exact_eigenvectors(adj)
            v2_k = eigvecs_k[:, 1]
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
            if not non_edges:
                break
            best_gap = -1
            best_e = None
            for i, j in non_edges:
                gap = (v2_k[i]-v2_k[j])**2
                if gap > best_gap:
                    best_gap = gap
                    best_e = (i, j)
            u, v = best_e
            adj[u, v] = 1; adj[v, u] = 1
        dl2_oracle.append(exact_lambda2_np(adj) - l2_init)

        # --- Random ---
        adj = adj_orig.copy()
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        rng.shuffle(non_edges)
        for i, j in non_edges[:n_edges]:
            adj[i,j] = 1; adj[j,i] = 1
        dl2_random.append(exact_lambda2_np(adj) - l2_init)

    print("=" * 70)
    print(f"TEST 4: Greedy ADD quality — {n_edges} edges added (n={n}, m={m})")
    print("=" * 70)
    print(f"  {'Method':<25s} | {'avg Δλ₂':>10s} | {'vs oracle':>10s}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10}")
    oracle_mean = np.mean(dl2_oracle)
    for label, vals in [
        ("Random", dl2_random),
        ("Stale greedy", dl2_stale),
        ("Corrected greedy", dl2_corrected),
        ("Oracle greedy", dl2_oracle),
    ]:
        m_val = np.mean(vals)
        pct = m_val / oracle_mean * 100 if oracle_mean > 0 else 0
        print(f"  {label:<25s} | {m_val:+10.4f} | {pct:9.1f}%")
    print()
    return np.mean(dl2_stale), np.mean(dl2_corrected), np.mean(dl2_oracle)


# ============================================================================
# Test 5: Scale test — does it work at larger n?
# ============================================================================
def test_scaling(n_values=[8, 12, 16, 24, 32], n_trials=100):
    """Test corrected greedy at different graph sizes."""
    rng = np.random.RandomState(42)

    print("=" * 70)
    print("TEST 5: Scaling — corrected greedy vs stale vs oracle")
    print("=" * 70)
    print(f"  {'n':>4s} {'m':>5s} {'n_add':>6s} | {'Stale':>8s} {'Corrected':>10s} {'Oracle':>8s} | {'S/O':>5s} {'C/O':>5s}")
    print(f"  {'-'*4} {'-'*5} {'-'*6} | {'-'*8} {'-'*10} {'-'*8} | {'-'*5} {'-'*5}")

    for n in n_values:
        m = n * (n - 1) // 4  # ~50% density
        n_edges = n  # add n edges (like delta=2 with n/2 effective)

        dl2_stale = []
        dl2_corrected = []
        dl2_oracle = []

        for _ in range(n_trials):
            adj = build_random_tree_initial(n, m, rng)
            l2_init = exact_lambda2_np(adj)

            eigvals, eigvecs = exact_eigenvectors(adj)
            v2_o = eigvecs[:, 1].copy()
            v3_o = eigvecs[:, 2].copy()
            lam2_o = eigvals[1]
            lam3_o = eigvals[2]

            # Stale
            adj_s = adj.copy()
            non_edges = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            scores = {e: (v2_o[e[0]]-v2_o[e[1]])**2 for e in non_edges}
            top = sorted(scores.keys(), key=lambda e: scores[e], reverse=True)
            for i, j in top[:n_edges]:
                adj_s[i,j] = 1; adj_s[j,i] = 1
            dl2_stale.append(exact_lambda2_np(adj_s) - l2_init)

            # Corrected
            adj_c = adj.copy()
            at = 0.0; l2 = lam2_o; l3 = lam3_o
            for _ in range(n_edges):
                ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_c[i,j]==0]
                if not ne: break
                best_s = -1; best_e = None
                for i, j in ne:
                    g = (v2_o[i]-v2_o[j]) + at*(v3_o[i]-v3_o[j])
                    s = g**2
                    if s > best_s: best_s = s; best_e = (i,j)
                u, v = best_e
                denom = l2 - l3
                if abs(denom) > 1e-12:
                    at += (v2_o[u]-v2_o[v])*(v3_o[u]-v3_o[v])/denom
                l2 += (v2_o[u]-v2_o[v])**2
                l3 += (v3_o[u]-v3_o[v])**2
                adj_c[u,v] = 1; adj_c[v,u] = 1
            dl2_corrected.append(exact_lambda2_np(adj_c) - l2_init)

            # Oracle
            adj_o = adj.copy()
            for _ in range(n_edges):
                ev, evec = exact_eigenvectors(adj_o)
                v2_k = evec[:, 1]
                ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_o[i,j]==0]
                if not ne: break
                best_g = -1; best_e = None
                for i, j in ne:
                    g = (v2_k[i]-v2_k[j])**2
                    if g > best_g: best_g = g; best_e = (i,j)
                u, v = best_e
                adj_o[u,v] = 1; adj_o[v,u] = 1
            dl2_oracle.append(exact_lambda2_np(adj_o) - l2_init)

        ms = np.mean(dl2_stale)
        mc = np.mean(dl2_corrected)
        mo = np.mean(dl2_oracle)
        s_pct = ms/mo*100 if mo > 0 else 0
        c_pct = mc/mo*100 if mo > 0 else 0
        print(f"  {n:4d} {m:5d} {n_edges:6d} | {ms:+8.3f} {mc:+10.3f} {mo:+8.3f} | {s_pct:4.0f}% {c_pct:4.0f}%")
    print()


if __name__ == "__main__":
    test_single_edge_accuracy(n=8, m=14)
    test_accumulated_accuracy(n=8, m=10, n_edges=8)
    test_scoring_quality(n=8, m=14, n_edges_placed=4)
    test_greedy_add_quality(n=8, m=14, n_edges=8)
    test_greedy_add_quality(n=8, m=10, n_edges=8)
    test_greedy_add_quality(n=8, m=18, n_edges=8)
    test_scaling()
