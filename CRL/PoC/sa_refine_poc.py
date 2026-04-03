#!/usr/bin/env python3
"""
PoC: Simulated Annealing REFINE — FV gradient + Metropolis acceptance.

=== Insight ===
FV provides the exact gradient of λ₂. REFINE can move in ANY direction
(add + remove edges). But greedy REFINE still gets trapped because it
never accepts worsening swaps.

Simulated Annealing fixes this:
  - FV gradient PROPOSES which swap to try (add high-gap, remove low-gap)
  - Metropolis criterion ACCEPTS/REJECTS: always accept improvements,
    accept worsenings with probability exp(Δλ₂ / T)
  - Temperature cools: explore broadly early, refine greedily late
  - Track best graph seen (SA might revisit worse states before finding better)

=== Key advantage over FV construction ===
FV can only ADD edges. Every mistake is permanent.
SA-REFINE can ADD and REMOVE. Mistakes are reversible.
SA lets it deliberately accept temporary λ₂ decreases to escape local optima.

=== Algorithm ===
For each swap iteration:
  1. Eigensolve current graph → v₂, λ₂
  2. PROPOSE: sample add-edge (FV-biased) + remove-edge (inverse-FV, avoid bridges)
  3. APPLY swap, eigensolve new graph → λ₂_new
  4. Δλ₂ = λ₂_new - λ₂
  5. ACCEPT if Δλ₂ > 0, else accept with prob exp(Δλ₂ / T)
  6. COOL temperature

Usage:
    python sa_refine_poc.py --n 16 --workers 8
    python sa_refine_poc.py --n 16,24 --workers 8
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import linalg
from multiprocessing import Pool, cpu_count

# ===== Baselines =====
_cache_path = Path(__file__).parent.parent.parent / "rl" / "baselines_cache.json"
_cache = {}
if _cache_path.exists():
    with open(_cache_path) as f:
        _cache = json.load(f)


def get_fv_baseline(n, m):
    return _cache.get(str(n), {}).get(str(m), {}).get('fv', 0.0)


def get_best_baseline(n, m):
    entry = _cache.get(str(n), {}).get(str(m), {})
    vals = [v for v in entry.values() if isinstance(v, (int, float)) and v > 0]
    return max(vals) if vals else 0.0


# ===== Graph utilities =====

def ring_plus_random(n, m, rng):
    """Ring + random edges to reach m total."""
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1
    current_m = n
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                non_edges.append((i, j))
    rng.shuffle(non_edges)
    for i, j in non_edges:
        if current_m >= m:
            break
        adj[i, j] = adj[j, i] = 1
        current_m += 1
    return adj


def algebraic_connectivity(adj):
    L = np.diag(adj.sum(axis=1)) - adj
    return float(linalg.eigvalsh(L)[1])


def find_bridges(adj, n):
    """Tarjan's bridge detection. Returns set of (i,j) with i<j."""
    bridges = set()
    disc = [-1] * n
    low = [-1] * n
    timer = [0]

    def dfs(u, parent):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in range(n):
            if adj[u, v] == 0:
                continue
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    import sys
    sys.setrecursionlimit(10000)
    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    return bridges


# ===== Greedy REFINE (baseline) =====

def greedy_refine(n, m, K, swap_frac, seed):
    """Greedy REFINE: K steps of batch add+remove, always reject worsening."""
    rng = np.random.default_rng(seed)
    adj = ring_plus_random(n, m, rng)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    num_swaps = max(1, int(m * swap_frac))

    for k in range(K):
        L = np.diag(adj.sum(axis=1)) - adj
        _, evecs = linalg.eigh(L)
        v2 = evecs[:, 1]
        scores = (v2[:, None] - v2[None, :]) ** 2

        # ADD phase: greedy, highest FV gap
        add_mask = (adj == 0) & upper_tri
        added = []
        for _ in range(num_swaps):
            if not add_mask.any():
                break
            idx = np.where(add_mask.ravel())[0]
            best = idx[np.argmax(scores.ravel()[idx])]
            i, j = best // n, best % n
            adj[i, j] = adj[j, i] = 1
            add_mask[i, j] = False
            added.append((i, j))

        if not added:
            continue

        # Recompute scores after additions
        L = np.diag(adj.sum(axis=1)) - adj
        _, evecs = linalg.eigh(L)
        v2 = evecs[:, 1]
        scores = (v2[:, None] - v2[None, :]) ** 2

        # REMOVE phase: greedy, lowest FV gap, avoid bridges
        bridges = find_bridges(adj, n)
        rem_mask = (adj > 0) & upper_tri
        for bi, bj in bridges:
            rem_mask[bi, bj] = False
        for ai, aj in added:
            a_min, a_max = min(ai, aj), max(ai, aj)
            rem_mask[a_min, a_max] = False

        removed = 0
        for _ in range(len(added)):
            if not rem_mask.any():
                break
            idx = np.where(rem_mask.ravel())[0]
            worst = idx[np.argmin(scores.ravel()[idx])]
            i, j = worst // n, worst % n
            adj[i, j] = adj[j, i] = 0
            rem_mask[i, j] = False
            removed += 1

        # Undo unmatched adds
        unmatched = len(added) - removed
        if unmatched > 0:
            for ai, aj in reversed(added[-unmatched:]):
                if adj[ai, aj] > 0:
                    adj[ai, aj] = adj[aj, ai] = 0

    return algebraic_connectivity(adj)


# ===== SA REFINE =====

def sa_refine(n, m, total_iters, T_start, T_end, proposal_tau, seed):
    """
    Simulated Annealing REFINE.

    Per iteration: propose one swap (add+remove) using FV-biased sampling,
    accept/reject via Metropolis criterion, geometric cooling.

    Returns: (best_λ₂, final_λ₂, accept_rate, worsen_accepts)
    """
    rng = np.random.default_rng(seed)
    adj = ring_plus_random(n, m, rng)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Initial eigensolve
    L = np.diag(adj.sum(axis=1)) - adj
    evals, evecs = linalg.eigh(L)
    v2 = evecs[:, 1]
    lam2 = float(evals[1])
    scores = (v2[:, None] - v2[None, :]) ** 2

    best_lam2 = lam2
    best_adj = adj.copy()

    accepts = 0
    worsen_accepts = 0
    total_proposals = 0

    for it in range(total_iters):
        # Geometric cooling
        progress = it / max(total_iters - 1, 1)
        T = T_start * (T_end / max(T_start, 1e-15)) ** progress

        # --- PROPOSE ADD: FV-biased non-edge ---
        add_mask = (adj == 0) & upper_tri
        if not add_mask.any():
            continue
        add_idx = np.where(add_mask.ravel())[0]
        add_scores = scores.ravel()[add_idx]

        if proposal_tau > 1e-10:
            logits = add_scores / proposal_tau
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            add_choice = add_idx[rng.choice(len(add_idx), p=probs)]
        else:
            add_choice = add_idx[np.argmax(add_scores)]
        ai, aj = add_choice // n, add_choice % n

        # --- PROPOSE REMOVE: inverse-FV-biased edge, avoid bridges ---
        bridges = find_bridges(adj, n)
        rem_mask = (adj > 0) & upper_tri
        for bi, bj in bridges:
            rem_mask[bi, bj] = False
        # Don't remove the edge we're about to add (it's not there yet)
        if not rem_mask.any():
            continue
        rem_idx = np.where(rem_mask.ravel())[0]
        rem_scores = scores.ravel()[rem_idx]

        if proposal_tau > 1e-10:
            # Minimize: negate scores
            logits = -rem_scores / proposal_tau
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            rem_choice = rem_idx[rng.choice(len(rem_idx), p=probs)]
        else:
            rem_choice = rem_idx[np.argmin(rem_scores)]
        ri, rj = rem_choice // n, rem_choice % n

        total_proposals += 1

        # --- APPLY SWAP ---
        adj[ai, aj] = adj[aj, ai] = 1
        adj[ri, rj] = adj[rj, ri] = 0

        # --- EVALUATE ---
        L_new = np.diag(adj.sum(axis=1)) - adj
        evals_new, evecs_new = linalg.eigh(L_new)
        lam2_new = float(evals_new[1])
        delta = lam2_new - lam2

        # --- METROPOLIS ACCEPTANCE ---
        if delta > 0:
            accept = True
        elif T > 1e-15:
            accept = rng.random() < np.exp(delta / T)
        else:
            accept = False

        if accept:
            accepts += 1
            if delta < -1e-8:
                worsen_accepts += 1
            lam2 = lam2_new
            v2 = evecs_new[:, 1]
            scores = (v2[:, None] - v2[None, :]) ** 2
            if lam2 > best_lam2:
                best_lam2 = lam2
                best_adj = adj.copy()
        else:
            # Undo swap
            adj[ai, aj] = adj[aj, ai] = 0
            adj[ri, rj] = adj[rj, ri] = 1

    accept_rate = accepts / max(total_proposals, 1)
    return best_lam2, lam2, accept_rate, worsen_accepts


# ===== Configuration =====

NUM_SEEDS = 5
K_REFINE = 20
SWAP_FRAC = 0.05
PROPOSAL_TAU = 0.01  # fixed proposal temperature for FV-biased sampling

SA_CONFIGS = {
    'sa_mild':     {'T_start': 0.01,  'T_end': 0.0001, 'mult': 2},
    'sa_medium':   {'T_start': 0.05,  'T_end': 0.001,  'mult': 2},
    'sa_hot':      {'T_start': 0.2,   'T_end': 0.001,  'mult': 2},
    'sa_very_hot': {'T_start': 1.0,   'T_end': 0.001,  'mult': 2},
    'sa_med_3x':   {'T_start': 0.05,  'T_end': 0.001,  'mult': 3},
    'sa_hot_3x':   {'T_start': 0.2,   'T_end': 0.001,  'mult': 3},
}


def _eval_config(args):
    n, m = args
    fv_db = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    results = {'n': n, 'm': m, 'fv_db': fv_db, 'best_bl': best_bl}

    # Greedy REFINE baseline
    results['greedy'] = max(
        greedy_refine(n, m, K_REFINE, SWAP_FRAC, seed=s)
        for s in range(NUM_SEEDS))

    # SA variants
    for name, cfg in SA_CONFIGS.items():
        total_iters = int(cfg['mult'] * m)
        sa_results = []
        for s in range(NUM_SEEDS):
            best_l2, final_l2, acc_rate, worsen = sa_refine(
                n, m, total_iters,
                cfg['T_start'], cfg['T_end'],
                PROPOSAL_TAU, seed=s)
            sa_results.append((best_l2, acc_rate, worsen))
        # Take best across seeds
        results[name] = max(r[0] for r in sa_results)
        results[f'{name}_acc'] = np.mean([r[1] for r in sa_results])
        results[f'{name}_worsen'] = np.mean([r[2] for r in sa_results])

    return results


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description="SA REFINE PoC: Simulated Annealing with FV gradient")
    parser.add_argument("--n", type=str, required=True)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8))
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]

    tasks = []
    for n in n_values:
        max_m = n * (n - 1) // 2
        for m in range(n + 1, max_m + 1):
            if get_best_baseline(n, m) > 0:
                tasks.append((n, m))

    sa_names = list(SA_CONFIGS.keys())
    print(f"SA REFINE PoC | n={args.n} | {len(tasks)} configs | "
          f"{args.workers} workers | {NUM_SEEDS} seeds")
    print(f"Greedy: K={K_REFINE}, frac={SWAP_FRAC}")
    print(f"SA: proposal_τ={PROPOSAL_TAU}, geometric cooling")
    for name, cfg in SA_CONFIGS.items():
        print(f"  {name}: T={cfg['T_start']}→{cfg['T_end']}, "
              f"iters={cfg['mult']}×m")
    print()

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    all_names = ['greedy'] + sa_names

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        max_m = n * (n - 1) // 2

        avg_fv = np.mean([r['fv_db'] for r in nr if r['fv_db'] > 0])
        avg_best = np.mean([r['best_bl'] for r in nr if r['best_bl'] > 0])

        print(f"\n{'=' * 100}")
        print(f"N = {n} ({len(nr)} configs)")
        print(f"{'=' * 100}")

        # --- Summary table ---
        print(f"\n  {'Method':<18} {'avg λ₂':>8} {'vs FV_db':>9} "
              f"{'W':>4} {'T':>3} {'vs best':>9} {'W':>4} "
              f"{'acc%':>6} {'↓acc':>5}")
        print(f"  {'-' * 75}")

        for name in all_names:
            avg = np.mean([r[name] for r in nr])
            r_fv = avg / avg_fv * 100 if avg_fv > 0 else 0
            r_best = avg / avg_best * 100 if avg_best > 0 else 0
            wins_fv = sum(1 for r in nr
                          if r[name] > r['fv_db'] + 1e-6 and r['fv_db'] > 0)
            ties_fv = sum(1 for r in nr
                          if abs(r[name] - r['fv_db']) < 1e-6
                          and r['fv_db'] > 0)
            wins_best = sum(1 for r in nr
                            if r[name] > r['best_bl'] + 1e-6
                            and r['best_bl'] > 0)

            if name in SA_CONFIGS:
                acc = np.mean([r[f'{name}_acc'] for r in nr]) * 100
                worsen = np.mean([r[f'{name}_worsen'] for r in nr])
                print(f"  {name:<18} {avg:8.3f} {r_fv:8.1f}% "
                      f"{wins_fv:4d} {ties_fv:3d} {r_best:8.1f}% "
                      f"{wins_best:4d} {acc:5.1f}% {worsen:5.1f}")
            else:
                print(f"  {name:<18} {avg:8.3f} {r_fv:8.1f}% "
                      f"{wins_fv:4d} {ties_fv:3d} {r_best:8.1f}% "
                      f"{wins_best:4d}     -     -")

        # --- Oracle ---
        oracle_vals = [max(r[name] for name in all_names) for r in nr]
        oracle_avg = np.mean(oracle_vals)
        oracle_wins = sum(1 for r, ov in zip(nr, oracle_vals)
                          if ov > r['best_bl'] + 1e-6 and r['best_bl'] > 0)
        print(f"\n  Oracle (per-config best): avg={oracle_avg:.3f} | "
              f"vs FV_db: {oracle_avg / avg_fv * 100:.1f}% | "
              f"vs best: {oracle_avg / avg_best * 100:.1f}% ({oracle_wins}W)")

        # --- SA vs greedy ---
        print(f"\n  SA improvements over greedy REFINE:")
        for name in sa_names:
            wins = sum(1 for r in nr
                       if r[name] > r['greedy'] + 1e-6)
            avg_gain = np.mean([r[name] - r['greedy'] for r in nr])
            print(f"    {name}: {wins}/{len(nr)} configs beat greedy "
                  f"(avg Δ={avg_gain:+.4f})")

        # --- Density breakdown ---
        print(f"\n  By density (best SA vs greedy vs FV_db):")
        bins = [
            ("sparse  <0.3", 0.0, 0.3),
            ("medium 0.3-0.6", 0.3, 0.6),
            ("dense   ≥0.6", 0.6, 1.01),
        ]
        for label, lo, hi in bins:
            sub = [r for r in nr if lo <= r['m'] / max_m < hi]
            if not sub:
                continue
            avg_greedy = np.mean([r['greedy'] for r in sub])
            avg_fv_sub = np.mean([r['fv_db'] for r in sub if r['fv_db'] > 0])
            avg_bl = np.mean([r['best_bl'] for r in sub if r['best_bl'] > 0])
            # Best SA for this density
            best_sa_name = None
            best_sa_avg = avg_greedy
            for name in sa_names:
                avg = np.mean([r[name] for r in sub])
                if avg > best_sa_avg:
                    best_sa_avg = avg
                    best_sa_name = name
            if best_sa_name:
                wins = sum(1 for r in sub
                           if r[best_sa_name] > r['greedy'] + 1e-6)
                print(f"    {label}: greedy={avg_greedy:.3f} "
                      f"best_SA({best_sa_name})={best_sa_avg:.3f} "
                      f"FV_db={avg_fv_sub:.3f} best_bl={avg_bl:.3f} "
                      f"({wins}W/{len(sub)})")
            else:
                print(f"    {label}: greedy={avg_greedy:.3f} "
                      f"(no SA improves) "
                      f"FV_db={avg_fv_sub:.3f} best_bl={avg_bl:.3f}")


if __name__ == "__main__":
    main()
