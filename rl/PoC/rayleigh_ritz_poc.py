#!/usr/bin/env python3
"""
PoC: Rayleigh-Ritz 2D Subspace Rotation for Autoregressive Edge Scoring

Instead of first-order perturbation (v₂ + α·v₃, which breaks at eigenvalue crossings),
we project L_new into span{v₂, v₃} and solve the EXACT 2×2 eigenproblem.

Key insight: Adding edge (u,v) updates L by rank-1: L_new = L + zz^T where z = e_u - e_v.
Project into V = [v₂, v₃]:

    L_sub = [[λ₂ + δ₂², δ₂·δ₃],
             [δ₂·δ₃,  λ₃ + δ₃²]]

where δ₂ = v₂[u] - v₂[v], δ₃ = v₃[u] - v₃[v].

Diagonalize this 2×2 analytically → new eigenvalues + rotation matrix R.
Update vectors: [v₂_new, v₃_new] = [v₂, v₃] @ R   (O(N) per edge)

This NATURALLY handles eigenvalue crossings — when λ₂ > λ₃, R just swaps them.

Comparison with first-order perturbation (perturbation_poc.py):
- First-order: divides by (λ₂ - λ₃) → blows up at crossings
- Rayleigh-Ritz: solves exact 2×2 → handles crossings gracefully
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
    """Normalize and fix sign ambiguity (make largest component positive)."""
    v = v / (np.linalg.norm(v) + 1e-15)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v


def solve_2x2_eig(a, b, c, d):
    """
    Analytical eigendecomposition of symmetric 2×2 matrix:
        [[a, b],
         [b, d]]

    Returns (lam1, lam2, R) where lam1 <= lam2 and R is 2×2 rotation matrix.
    Columns of R are eigenvectors: R[:,0] for lam1, R[:,1] for lam2.

    O(1) computation — no library calls needed.
    """
    trace = a + d
    det = a * d - b * b
    disc = max(0.0, trace * trace - 4.0 * det)
    sqrt_disc = np.sqrt(disc)

    lam1 = (trace - sqrt_disc) / 2.0  # smaller eigenvalue
    lam2 = (trace + sqrt_disc) / 2.0  # larger eigenvalue

    # Eigenvectors of 2×2 symmetric matrix
    if abs(b) < 1e-15:
        # Already diagonal
        if a <= d:
            R = np.eye(2)
        else:
            R = np.array([[0.0, 1.0], [1.0, 0.0]])
    else:
        # First eigenvector (for lam1)
        v1 = np.array([b, lam1 - a])
        v1 /= np.linalg.norm(v1)
        # Second eigenvector (for lam2) — orthogonal
        v2 = np.array([-v1[1], v1[0]])
        R = np.column_stack([v1, v2])

    return lam1, lam2, R


def rayleigh_ritz_update(v2, v3, lam2, lam3, u, v_node):
    """
    Rayleigh-Ritz 2D subspace update after adding edge (u, v_node).

    Projects L_new into span{v₂, v₃}, solves exact 2×2 eigenproblem,
    rotates v₂, v₃ accordingly.

    Args:
        v2, v3: current Fiedler and 3rd eigenvectors (N,)
        lam2, lam3: current eigenvalues
        u, v_node: edge endpoints being added

    Returns:
        v2_new, v3_new: updated eigenvectors (N,)
        lam2_new, lam3_new: updated eigenvalues

    Complexity: O(N) for vector rotation, O(1) for eigenvalues.
    """
    delta2 = v2[u] - v2[v_node]
    delta3 = v3[u] - v3[v_node]

    # Build 2×2 projected matrix
    a = lam2 + delta2 * delta2
    b = delta2 * delta3
    d = lam3 + delta3 * delta3

    # Solve 2×2 eigenproblem analytically
    lam_lo, lam_hi, R = solve_2x2_eig(a, b, a_d_placeholder=0, d=d)
    # Fix: need to pass correct args
    lam_lo, lam_hi, R = solve_2x2_eig(a, b, None, d)

    # The smaller eigenvalue is the new λ₂ (second-smallest of full Laplacian)
    # Convention: λ₂ < λ₃, so lam_lo → λ₂, lam_hi → λ₃
    lam2_new = lam_lo
    lam3_new = lam_hi

    # Rotate full N-dim vectors: [v2_new, v3_new] = [v2, v3] @ R
    v2_new = R[0, 0] * v2 + R[1, 0] * v3  # O(N)
    v3_new = R[0, 1] * v2 + R[1, 1] * v3  # O(N)

    return v2_new, v3_new, lam2_new, lam3_new


# Fix the solve_2x2_eig signature issue
def rayleigh_ritz_update(v2, v3, lam2, lam3, u, v_node):
    """
    Rayleigh-Ritz 2D subspace update after adding edge (u, v_node).
    """
    delta2 = v2[u] - v2[v_node]
    delta3 = v3[u] - v3[v_node]

    # Build 2×2 projected matrix [[a, b], [b, d]]
    a = lam2 + delta2 * delta2
    b = delta2 * delta3
    d = lam3 + delta3 * delta3

    # Analytical 2×2 eigensolver
    trace = a + d
    det = a * d - b * b
    disc = max(0.0, trace * trace - 4.0 * det)
    sqrt_disc = np.sqrt(disc)

    lam2_new = (trace - sqrt_disc) / 2.0  # smaller
    lam3_new = (trace + sqrt_disc) / 2.0  # larger

    # Eigenvectors of 2×2
    if abs(b) < 1e-15:
        if a <= d:
            # No rotation needed
            return v2.copy(), v3.copy(), lam2_new, lam3_new
        else:
            # Swap v₂ and v₃
            return v3.copy(), v2.copy(), lam2_new, lam3_new
    else:
        # Eigenvector for lam2_new (smaller eigenvalue)
        r1 = np.array([b, lam2_new - a])
        r1 /= np.linalg.norm(r1)
        r2 = np.array([-r1[1], r1[0]])  # orthogonal

        # Rotate: v2_new = r1[0]*v2 + r1[1]*v3, v3_new = r2[0]*v2 + r2[1]*v3
        v2_new = r1[0] * v2 + r1[1] * v3  # O(N)
        v3_new = r2[0] * v2 + r2[1] * v3  # O(N)

        return v2_new, v3_new, lam2_new, lam3_new


# ============================================================================
# Test 1: Single edge — Rayleigh-Ritz accuracy vs first-order perturbation
# ============================================================================
def test_single_edge(n=8, m=14, n_trials=500):
    """Compare single-edge accuracy: RR vs first-order perturbation."""
    rng = np.random.RandomState(42)

    cos_rr = []
    cos_pert = []
    lam_err_rr = []
    lam_err_pert = []

    for _ in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)
        v2 = eigvecs[:, 1].copy()
        v3 = eigvecs[:, 2].copy()
        lam2 = eigvals[1]
        lam3 = eigvals[2]

        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        if not non_edges:
            continue
        u, v = non_edges[rng.randint(len(non_edges))]

        # --- Rayleigh-Ritz ---
        v2_rr, v3_rr, lam2_rr, lam3_rr = rayleigh_ritz_update(v2, v3, lam2, lam3, u, v)
        v2_rr = normalize_eigenvector(v2_rr)

        # --- First-order perturbation ---
        delta2 = v2[u] - v2[v]
        delta3 = v3[u] - v3[v]
        denom = lam2 - lam3
        alpha = delta2 * delta3 / denom if abs(denom) > 1e-12 else 0.0
        v2_pert = normalize_eigenvector(v2 + alpha * v3)
        lam2_pert = lam2 + delta2**2

        # --- Ground truth ---
        adj[u, v] = 1; adj[v, u] = 1
        eigvals_new, eigvecs_new = exact_eigenvectors(adj)
        v2_true = normalize_eigenvector(eigvecs_new[:, 1])
        lam2_true = eigvals_new[1]

        cos_rr.append(abs(np.dot(v2_rr, v2_true)))
        cos_pert.append(abs(np.dot(v2_pert, v2_true)))
        lam_err_rr.append(abs(lam2_rr - lam2_true))
        lam_err_pert.append(abs(lam2_pert - lam2_true))

    print("=" * 70)
    print(f"TEST 1: Single-edge accuracy (n={n}, m={m}, {n_trials} trials)")
    print("=" * 70)
    print(f"  {'Metric':<30s} | {'First-order':>12s} | {'Rayleigh-Ritz':>13s}")
    print(f"  {'-'*30} | {'-'*12} | {'-'*13}")
    print(f"  {'v₂ cosine sim (mean)':<30s} | {np.mean(cos_pert):12.4f} | {np.mean(cos_rr):13.4f}")
    print(f"  {'v₂ cosine sim > 0.99':<30s} | {sum(1 for c in cos_pert if c > 0.99)/len(cos_pert)*100:11.1f}% | {sum(1 for c in cos_rr if c > 0.99)/len(cos_rr)*100:12.1f}%")
    print(f"  {'v₂ cosine sim > 0.95':<30s} | {sum(1 for c in cos_pert if c > 0.95)/len(cos_pert)*100:11.1f}% | {sum(1 for c in cos_rr if c > 0.95)/len(cos_rr)*100:12.1f}%")
    print(f"  {'λ₂ abs error (mean)':<30s} | {np.mean(lam_err_pert):12.4f} | {np.mean(lam_err_rr):13.4f}")
    print()


# ============================================================================
# Test 2: Accumulated edges — the critical test
# ============================================================================
def test_accumulated(n=8, m=10, n_edges=8, n_trials=200):
    """Track accuracy of RR vs first-order perturbation over many sequential edges."""
    rng = np.random.RandomState(42)

    results_rr = {k: [] for k in range(1, n_edges + 1)}
    results_pert = {k: [] for k in range(1, n_edges + 1)}
    lam_results_rr = {k: [] for k in range(1, n_edges + 1)}
    lam_results_pert = {k: [] for k in range(1, n_edges + 1)}

    for _ in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)

        # RR state — mutable
        v2_rr = eigvecs[:, 1].copy()
        v3_rr = eigvecs[:, 2].copy()
        lam2_rr = eigvals[1]
        lam3_rr = eigvals[2]

        # First-order state — accumulates scalar α
        v2_orig = eigvecs[:, 1].copy()
        v3_orig = eigvecs[:, 2].copy()
        alpha_total = 0.0
        lam2_pert = eigvals[1]
        lam3_pert = eigvals[2]

        for k in range(1, n_edges + 1):
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
            if not non_edges:
                break
            u, v = non_edges[rng.randint(len(non_edges))]

            # --- RR update ---
            v2_rr, v3_rr, lam2_rr, lam3_rr = rayleigh_ritz_update(
                v2_rr, v3_rr, lam2_rr, lam3_rr, u, v
            )

            # --- First-order update ---
            delta2 = v2_orig[u] - v2_orig[v]
            delta3 = v3_orig[u] - v3_orig[v]
            denom = lam2_pert - lam3_pert
            if abs(denom) > 1e-12:
                alpha_total += delta2 * delta3 / denom
            lam2_pert += delta2**2
            lam3_pert += delta3**2

            # Apply edge
            adj[u, v] = 1; adj[v, u] = 1

            # Ground truth
            eigvals_true, eigvecs_true = exact_eigenvectors(adj)
            v2_true = normalize_eigenvector(eigvecs_true[:, 1])
            lam2_true = eigvals_true[1]

            # RR accuracy
            v2_rr_n = normalize_eigenvector(v2_rr)
            results_rr[k].append(abs(np.dot(v2_rr_n, v2_true)))
            lam_results_rr[k].append(abs(lam2_rr - lam2_true))

            # First-order accuracy
            v2_pert_n = normalize_eigenvector(v2_orig + alpha_total * v3_orig)
            results_pert[k].append(abs(np.dot(v2_pert_n, v2_true)))
            lam_results_pert[k].append(abs(lam2_pert - lam2_true))

    print("=" * 70)
    print(f"TEST 2: Accumulated accuracy (n={n}, m={m}, {n_edges} edges)")
    print("=" * 70)
    print(f"  {'edges':>5s} | {'RR cos':>8s} {'RR >0.9':>8s} {'RR λ₂err':>9s} | {'Pert cos':>9s} {'Pert >0.9':>10s} {'Pert λ₂err':>11s}")
    print(f"  {'-'*5} | {'-'*8} {'-'*8} {'-'*9} | {'-'*9} {'-'*10} {'-'*11}")
    for k in range(1, n_edges + 1):
        if results_rr[k] and results_pert[k]:
            rr_cos = np.mean(results_rr[k])
            rr_90 = sum(1 for c in results_rr[k] if c > 0.90) / len(results_rr[k]) * 100
            rr_lam = np.mean(lam_results_rr[k])
            p_cos = np.mean(results_pert[k])
            p_90 = sum(1 for c in results_pert[k] if c > 0.90) / len(results_pert[k]) * 100
            p_lam = np.mean(lam_results_pert[k])
            print(f"  {k:5d} | {rr_cos:8.4f} {rr_90:7.1f}% {rr_lam:9.4f} | {p_cos:9.4f} {p_90:9.1f}% {p_lam:11.4f}")
    print()


# ============================================================================
# Test 3: Greedy ADD quality — the bottom line
# ============================================================================
def test_greedy_add(n=8, m=14, n_edges=8, n_trials=300):
    """
    Compare total Δλ₂ from greedy ADD using:
    a) Stale: score all with original v₂
    b) First-order corrected: v₂ + α·v₃
    c) Rayleigh-Ritz corrected: 2D subspace rotation after each edge
    d) Oracle: recompute exact eigenvectors after each edge
    e) Random: baseline
    """
    rng = np.random.RandomState(42)

    dl2_stale = []
    dl2_pert = []
    dl2_rr = []
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

        # --- Stale greedy: score all with original v₂, add top-k ---
        adj = adj_orig.copy()
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        scores = {(i,j): (v2_orig[i]-v2_orig[j])**2 for i,j in non_edges}
        top = sorted(scores.keys(), key=lambda e: scores[e], reverse=True)
        for i, j in top[:n_edges]:
            adj[i,j] = 1; adj[j,i] = 1
        dl2_stale.append(exact_lambda2_np(adj) - l2_init)

        # --- First-order corrected greedy ---
        adj = adj_orig.copy()
        alpha_total = 0.0
        lam2_p = lam2_orig; lam3_p = lam3_orig
        for _ in range(n_edges):
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            if not ne: break
            best_s = -1; best_e = None
            for i, j in ne:
                gap = (v2_orig[i]-v2_orig[j]) + alpha_total*(v3_orig[i]-v3_orig[j])
                s = gap**2
                if s > best_s: best_s = s; best_e = (i,j)
            u, v = best_e
            denom = lam2_p - lam3_p
            if abs(denom) > 1e-12:
                alpha_total += (v2_orig[u]-v2_orig[v])*(v3_orig[u]-v3_orig[v])/denom
            lam2_p += (v2_orig[u]-v2_orig[v])**2
            lam3_p += (v3_orig[u]-v3_orig[v])**2
            adj[u,v] = 1; adj[v,u] = 1
        dl2_pert.append(exact_lambda2_np(adj) - l2_init)

        # --- Rayleigh-Ritz corrected greedy ---
        adj = adj_orig.copy()
        v2_rr = v2_orig.copy()
        v3_rr = v3_orig.copy()
        lam2_rr = lam2_orig
        lam3_rr = lam3_orig
        for _ in range(n_edges):
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            if not ne: break
            # Score using current RR-tracked v₂
            best_s = -1; best_e = None
            for i, j in ne:
                gap = (v2_rr[i] - v2_rr[j])**2
                if gap > best_s: best_s = gap; best_e = (i,j)
            u, v = best_e
            # Update via Rayleigh-Ritz
            v2_rr, v3_rr, lam2_rr, lam3_rr = rayleigh_ritz_update(
                v2_rr, v3_rr, lam2_rr, lam3_rr, u, v
            )
            adj[u,v] = 1; adj[v,u] = 1
        dl2_rr.append(exact_lambda2_np(adj) - l2_init)

        # --- Oracle greedy ---
        adj = adj_orig.copy()
        for _ in range(n_edges):
            ev, evec = exact_eigenvectors(adj)
            v2_k = evec[:, 1]
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            if not ne: break
            best_g = -1; best_e = None
            for i, j in ne:
                g = (v2_k[i]-v2_k[j])**2
                if g > best_g: best_g = g; best_e = (i,j)
            u, v = best_e
            adj[u,v] = 1; adj[v,u] = 1
        dl2_oracle.append(exact_lambda2_np(adj) - l2_init)

        # --- Random ---
        adj = adj_orig.copy()
        ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
        rng.shuffle(ne)
        for i, j in ne[:n_edges]:
            adj[i,j] = 1; adj[j,i] = 1
        dl2_random.append(exact_lambda2_np(adj) - l2_init)

    print("=" * 70)
    print(f"TEST 3: Greedy ADD quality — {n_edges} edges (n={n}, m={m})")
    print("=" * 70)
    oracle_mean = np.mean(dl2_oracle)
    print(f"  {'Method':<25s} | {'avg Δλ₂':>10s} | {'vs oracle':>10s}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10}")
    for label, vals in [
        ("Random", dl2_random),
        ("Stale greedy", dl2_stale),
        ("First-order corrected", dl2_pert),
        ("Rayleigh-Ritz corrected", dl2_rr),
        ("Oracle greedy", dl2_oracle),
    ]:
        m_val = np.mean(vals)
        pct = m_val / oracle_mean * 100 if oracle_mean > 0 else 0
        print(f"  {label:<25s} | {m_val:+10.4f} | {pct:9.1f}%")
    print()
    return {
        'stale': np.mean(dl2_stale),
        'pert': np.mean(dl2_pert),
        'rr': np.mean(dl2_rr),
        'oracle': np.mean(dl2_oracle),
        'random': np.mean(dl2_random),
    }


# ============================================================================
# Test 4: Scaling — does RR hold at larger n?
# ============================================================================
def test_scaling(n_values=[8, 12, 16, 24, 32], n_trials=100):
    """Test RR corrected greedy at different graph sizes."""
    rng = np.random.RandomState(42)

    print("=" * 70)
    print("TEST 4: Scaling — Rayleigh-Ritz vs stale vs first-order vs oracle")
    print("=" * 70)
    print(f"  {'n':>4s} {'m':>5s} {'#add':>5s} | {'Stale':>7s} {'1st-ord':>8s} {'RR':>7s} {'Oracle':>7s} | {'S/O':>5s} {'P/O':>5s} {'RR/O':>5s}")
    print(f"  {'-'*4} {'-'*5} {'-'*5} | {'-'*7} {'-'*8} {'-'*7} {'-'*7} | {'-'*5} {'-'*5} {'-'*5}")

    for n in n_values:
        m = n * (n - 1) // 4  # ~50% density
        n_edges = n

        dl2_s = []; dl2_p = []; dl2_rr = []; dl2_o = []

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
            ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            sc = {e: (v2_o[e[0]]-v2_o[e[1]])**2 for e in ne}
            top = sorted(sc.keys(), key=lambda e: sc[e], reverse=True)
            for i, j in top[:n_edges]:
                adj_s[i,j] = 1; adj_s[j,i] = 1
            dl2_s.append(exact_lambda2_np(adj_s) - l2_init)

            # First-order
            adj_p = adj.copy()
            at = 0.0; l2p = lam2_o; l3p = lam3_o
            for _ in range(n_edges):
                ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_p[i,j]==0]
                if not ne: break
                bs = -1; be = None
                for i, j in ne:
                    g = (v2_o[i]-v2_o[j]) + at*(v3_o[i]-v3_o[j])
                    s = g**2
                    if s > bs: bs = s; be = (i,j)
                u, v = be
                dn = l2p - l3p
                if abs(dn) > 1e-12:
                    at += (v2_o[u]-v2_o[v])*(v3_o[u]-v3_o[v])/dn
                l2p += (v2_o[u]-v2_o[v])**2
                l3p += (v3_o[u]-v3_o[v])**2
                adj_p[u,v] = 1; adj_p[v,u] = 1
            dl2_p.append(exact_lambda2_np(adj_p) - l2_init)

            # Rayleigh-Ritz
            adj_r = adj.copy()
            v2r = v2_o.copy(); v3r = v3_o.copy()
            l2r = lam2_o; l3r = lam3_o
            for _ in range(n_edges):
                ne = [(i,j) for i in range(n) for j in range(i+1,n) if adj_r[i,j]==0]
                if not ne: break
                bs = -1; be = None
                for i, j in ne:
                    g = (v2r[i]-v2r[j])**2
                    s = g
                    if s > bs: bs = s; be = (i,j)
                u, v = be
                v2r, v3r, l2r, l3r = rayleigh_ritz_update(v2r, v3r, l2r, l3r, u, v)
                adj_r[u,v] = 1; adj_r[v,u] = 1
            dl2_rr.append(exact_lambda2_np(adj_r) - l2_init)

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

        ms = np.mean(dl2_s); mp = np.mean(dl2_p)
        mr = np.mean(dl2_rr); mo = np.mean(dl2_o)
        sp = ms/mo*100 if mo>0 else 0
        pp = mp/mo*100 if mo>0 else 0
        rp = mr/mo*100 if mo>0 else 0
        print(f"  {n:4d} {m:5d} {n_edges:5d} | {ms:+7.3f} {mp:+8.3f} {mr:+7.3f} {mo:+7.3f} | {sp:4.0f}% {pp:4.0f}% {rp:4.0f}%")
    print()


# ============================================================================
# Test 5: Eigenvalue crossing analysis — does RR handle it?
# ============================================================================
def test_eigenvalue_crossing(n=8, m=10, n_edges=8, n_trials=300):
    """
    Track how well RR handles eigenvalue crossings vs first-order perturbation.
    Measures accuracy specifically when λ₂ crosses λ₃.
    """
    rng = np.random.RandomState(42)

    # Track accuracy conditional on crossing
    crossed_rr = []
    crossed_pert = []
    not_crossed_rr = []
    not_crossed_pert = []
    n_crossings = 0
    n_total = 0

    for _ in range(n_trials):
        adj = build_random_tree_initial(n, m, rng)
        eigvals, eigvecs = exact_eigenvectors(adj)

        v2_rr = eigvecs[:, 1].copy()
        v3_rr = eigvecs[:, 2].copy()
        lam2_rr_val = eigvals[1]
        lam3_rr_val = eigvals[2]

        v2_orig = eigvecs[:, 1].copy()
        v3_orig = eigvecs[:, 2].copy()
        alpha_total = 0.0
        lam2_pert = eigvals[1]
        lam3_pert = eigvals[2]

        prev_gap = eigvals[2] - eigvals[1]

        for k in range(n_edges):
            non_edges = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]==0]
            if not non_edges: break

            # Greedy with stale v₂ (worst case for staleness)
            best_g = -1; best_e = None
            for i, j in non_edges:
                g = (v2_orig[i]-v2_orig[j])**2
                if g > best_g: best_g = g; best_e = (i,j)
            u, v = best_e

            # Update RR
            v2_rr, v3_rr, lam2_rr_val, lam3_rr_val = rayleigh_ritz_update(
                v2_rr, v3_rr, lam2_rr_val, lam3_rr_val, u, v
            )

            # Update first-order
            d2 = v2_orig[u] - v2_orig[v]
            d3 = v3_orig[u] - v3_orig[v]
            dn = lam2_pert - lam3_pert
            if abs(dn) > 1e-12:
                alpha_total += d2*d3/dn
            lam2_pert += d2**2
            lam3_pert += d3**2

            adj[u,v] = 1; adj[v,u] = 1

            # True
            eigvals_true, eigvecs_true = exact_eigenvectors(adj)
            v2_true = normalize_eigenvector(eigvecs_true[:, 1])
            curr_gap = eigvals_true[2] - eigvals_true[1]

            v2_rr_n = normalize_eigenvector(v2_rr)
            v2_pert_n = normalize_eigenvector(v2_orig + alpha_total * v3_orig)

            cos_rr = abs(np.dot(v2_rr_n, v2_true))
            cos_pert = abs(np.dot(v2_pert_n, v2_true))

            n_total += 1
            # Detect crossing: gap changed sign or nearly touched
            if curr_gap < prev_gap * 0.3 or curr_gap < 0.05:
                n_crossings += 1
                crossed_rr.append(cos_rr)
                crossed_pert.append(cos_pert)
            else:
                not_crossed_rr.append(cos_rr)
                not_crossed_pert.append(cos_pert)

            prev_gap = curr_gap

    print("=" * 70)
    print(f"TEST 5: Eigenvalue crossing handling (n={n}, m={m})")
    print("=" * 70)
    print(f"  Total edge additions: {n_total}")
    print(f"  Near-crossing events: {n_crossings} ({n_crossings/n_total*100:.1f}%)")
    print()
    if crossed_rr:
        print(f"  AT/NEAR crossings:")
        print(f"    RR cosine sim:     {np.mean(crossed_rr):.4f} (>0.9: {sum(1 for c in crossed_rr if c>0.9)/len(crossed_rr)*100:.0f}%)")
        print(f"    1st-ord cosine sim: {np.mean(crossed_pert):.4f} (>0.9: {sum(1 for c in crossed_pert if c>0.9)/len(crossed_pert)*100:.0f}%)")
    if not_crossed_rr:
        print(f"  Away from crossings:")
        print(f"    RR cosine sim:     {np.mean(not_crossed_rr):.4f} (>0.9: {sum(1 for c in not_crossed_rr if c>0.9)/len(not_crossed_rr)*100:.0f}%)")
        print(f"    1st-ord cosine sim: {np.mean(not_crossed_pert):.4f} (>0.9: {sum(1 for c in not_crossed_pert if c>0.9)/len(not_crossed_pert)*100:.0f}%)")
    print()


if __name__ == "__main__":
    test_single_edge(n=8, m=14)
    test_accumulated(n=8, m=10, n_edges=8)
    test_greedy_add(n=8, m=14, n_edges=8)
    test_greedy_add(n=8, m=10, n_edges=8)
    test_greedy_add(n=8, m=18, n_edges=8)
    test_scaling()
    test_eigenvalue_crossing(n=8, m=10, n_edges=8)
