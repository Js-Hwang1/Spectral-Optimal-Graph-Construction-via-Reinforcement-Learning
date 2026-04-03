#!/usr/bin/env python3
"""
PoC: Second-order perturbation greedy vs FV (first-order greedy).

Tests whether the second-order correction to Δλ₂ actually helps
pick better edges than pure Fiedler-gap scoring.

score_add(a,b) = (v₂[a]-v₂[b])² × [1 - Σ_{k≥3} (v_k[a]-v_k[b])² / (λ_k - λ₂)]
score_rem(c,d) = (v₂[c]-v₂[d])² × [1 + Σ_{k≥3} (v_k[c]-v_k[d])² / (λ_k - λ₂)]
  (remove = minimize damage)

Compares:
  - FV greedy (first-order only): max (v₂[i]-v₂[j])²
  - SO greedy (second-order):     max score_add, min score_rem
  - Both use same init, same number of swaps, same bridge mask.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.gnm_env import build_ring_random_initial, algebraic_connectivity
from utils.spectral import exact_lambda2_np, find_bridges


def full_eigen(adj):
    """Full eigendecomposition of Laplacian. Returns (eigenvalues, eigenvectors)."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals, vecs = np.linalg.eigh(L)
    return vals, vecs


def greedy_swap_fv(adj, n, m, num_swaps, rng):
    """First-order FV greedy: add max Fiedler gap, remove min Fiedler gap."""
    adj = adj.copy()
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    for _ in range(num_swaps):
        vals, vecs = full_eigen(adj)
        v2 = vecs[:, 1]

        # Score all pairs by Fiedler gap
        fiedler_gap = (v2[:, None] - v2[None, :]) ** 2

        # ADD: max Fiedler gap among non-edges
        add_mask = (adj == 0) & upper_tri
        if not add_mask.any():
            break
        add_scores = np.where(add_mask, fiedler_gap, -np.inf)
        flat_add = add_scores.reshape(-1)
        best_add = int(np.argmax(flat_add))
        ai, aj = best_add // n, best_add % n

        # Add edge
        adj[ai, aj] = 1.0
        adj[aj, ai] = 1.0

        # REMOVE: min Fiedler gap among edges (excluding bridges and just-added)
        bridges = find_bridges(adj)
        bridge_mask = np.zeros((n, n), dtype=bool)
        for u, v in bridges:
            bridge_mask[u, v] = True
            bridge_mask[v, u] = True

        rem_mask = (adj > 0) & upper_tri & ~bridge_mask
        ai2, aj2 = min(ai, aj), max(ai, aj)
        rem_mask[ai2, aj2] = False

        if not rem_mask.any():
            # Undo add
            adj[ai, aj] = 0
            adj[aj, ai] = 0
            break

        rem_scores = np.where(rem_mask, fiedler_gap, np.inf)
        flat_rem = rem_scores.reshape(-1)
        best_rem = int(np.argmin(flat_rem))
        ri, rj = best_rem // n, best_rem % n

        # Remove edge
        adj[ri, rj] = 0
        adj[rj, ri] = 0

    return algebraic_connectivity(adj)


def greedy_swap_so(adj, n, m, num_swaps, rng):
    """Second-order greedy: use perturbation correction with v₃..v_k."""
    adj = adj.copy()
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    for _ in range(num_swaps):
        vals, vecs = full_eigen(adj)
        lam2 = vals[1]

        # Use eigenvectors 2..min(n, 8) for second-order correction
        n_eig = min(n - 1, 8)

        # Compute Fiedler gap (first-order)
        v2 = vecs[:, 1]
        fiedler_gap = (v2[:, None] - v2[None, :]) ** 2  # (n, n)

        # Compute second-order correction factor
        # correction(a,b) = Σ_{k≥3} (v_k[a]-v_k[b])² / (λ_k - λ₂)
        correction = np.zeros((n, n))
        for k in range(2, n_eig + 1):
            vk = vecs[:, k]
            lam_k = vals[k]
            gap_k = lam_k - lam2
            if gap_k < 1e-10:
                gap_k = 1e-10  # avoid division by zero for degenerate case
            vk_gap = (vk[:, None] - vk[None, :]) ** 2
            correction += vk_gap / gap_k

        # ADD score: fiedler_gap × (1 - correction)
        # Higher = better add
        add_score_so = fiedler_gap * (1.0 - correction)

        add_mask = (adj == 0) & upper_tri
        if not add_mask.any():
            break
        add_scores = np.where(add_mask, add_score_so, -np.inf)
        flat_add = add_scores.reshape(-1)
        best_add = int(np.argmax(flat_add))
        ai, aj = best_add // n, best_add % n

        # Add edge
        adj[ai, aj] = 1.0
        adj[aj, ai] = 1.0

        # REMOVE score: fiedler_gap × (1 + correction)
        # Lower = less damage = better remove
        rem_score_so = fiedler_gap * (1.0 + correction)

        bridges = find_bridges(adj)
        bridge_mask = np.zeros((n, n), dtype=bool)
        for u, v in bridges:
            bridge_mask[u, v] = True
            bridge_mask[v, u] = True

        rem_mask = (adj > 0) & upper_tri & ~bridge_mask
        ai2, aj2 = min(ai, aj), max(ai, aj)
        rem_mask[ai2, aj2] = False

        if not rem_mask.any():
            adj[ai, aj] = 0
            adj[aj, ai] = 0
            break

        rem_scores = np.where(rem_mask, rem_score_so, np.inf)
        flat_rem = rem_scores.reshape(-1)
        best_rem = int(np.argmin(flat_rem))
        ri, rj = best_rem // n, best_rem % n

        adj[ri, rj] = 0
        adj[rj, ri] = 0

    return algebraic_connectivity(adj)


def load_baselines(n_values):
    """Load baselines from cache."""
    cache_path = Path(__file__).parent.parent / 'baselines_cache.json'
    if not cache_path.exists():
        print("No baselines_cache.json found. Run eval_refine.py first.")
        return {}
    with open(cache_path) as f:
        return json.load(f)


def main():
    n_values = [8, 10, 16, 24]
    num_trials = 5
    num_swaps_mult = 3  # 3*m swaps per trial

    cache = load_baselines(n_values)

    print("=" * 100)
    print("Second-Order Perturbation PoC: FV-greedy vs SO-greedy (swap-based)")
    print("=" * 100)

    for n in n_values:
        n_key = str(n)
        if n_key not in cache:
            print(f"\nn={n}: no baselines cached, skipping")
            continue

        n_data = cache[n_key]
        max_m = n * (n - 1) // 2
        min_m = n - 1

        # Sample ~15 m values across the density range
        all_m = list(range(min_m, max_m + 1))
        if len(all_m) > 15:
            step = max(1, len(all_m) // 15)
            m_samples = all_m[::step]
            if all_m[-1] not in m_samples:
                m_samples.append(all_m[-1])
        else:
            m_samples = all_m

        print(f"\n{'='*100}")
        print(f"n={n} ({len(m_samples)} m values, {num_trials} trials each)")
        print(f"{'='*100}")
        print(f"{'(n,m)':>9}  {'DB-FV':>9} {'DB-ER':>9} {'FV-swap':>9} {'SO-swap':>9} "
              f"{'SO-FV':>9} {'SO>FV?':>6} {'SO>DB?':>6}")
        print("-" * 100)

        fv_wins = 0
        so_wins = 0
        so_beat_db = 0
        total = 0

        for m in m_samples:
            m_key = str(m)
            bl = n_data.get(m_key, {})
            if not bl:
                continue

            db_fv = bl.get('fv', 0) or 0
            db_er = bl.get('er', 0) or 0
            db_best = max(db_fv, db_er,
                         bl.get('sw_025', 0) or 0,
                         bl.get('sw_050', 0) or 0,
                         bl.get('sw_075', 0) or 0)

            fv_results = []
            so_results = []
            num_swaps = num_swaps_mult * m

            for trial in range(num_trials):
                rng = np.random.RandomState(trial)
                adj = build_ring_random_initial(n, m, rng)

                fv_l2 = greedy_swap_fv(adj.copy(), n, m, num_swaps, rng)
                so_l2 = greedy_swap_so(adj.copy(), n, m, num_swaps, rng)

                fv_results.append(fv_l2)
                so_results.append(so_l2)

            fv_mean = np.mean(fv_results)
            so_mean = np.mean(so_results)
            diff = so_mean - fv_mean
            so_better = so_mean > fv_mean + 1e-6
            so_beats_db = so_mean > db_best + 1e-6

            if so_better:
                so_wins += 1
            elif fv_mean > so_mean + 1e-6:
                fv_wins += 1

            if so_beats_db:
                so_beat_db += 1
            total += 1

            label = f"({n},{m})"
            so_fv_tag = "YES" if so_better else ""
            so_db_tag = "YES" if so_beats_db else ""
            print(f"{label:>9}  {db_fv:>9.4f} {db_er:>9.4f} {fv_mean:>9.4f} "
                  f"{so_mean:>9.4f} {diff:>+9.4f} {so_fv_tag:>6} {so_db_tag:>6}")

        print(f"\nn={n} summary: SO beats FV-swap in {so_wins}/{total}, "
              f"FV-swap beats SO in {fv_wins}/{total}, "
              f"SO beats DB-best in {so_beat_db}/{total}")

    print(f"\n{'='*100}")
    print("Done.")


if __name__ == "__main__":
    main()
