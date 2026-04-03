#!/usr/bin/env python3
"""
PoC: REFINE + Temperature — Batch rewiring with softmax-sampled FV gradient.

Combines two instruments:
  1. REFINE structure: K Lanczos steps × batch add+remove rewiring
  2. Temperature: softmax sampling on Fiedler gap scores (not argmax)

This mimics what a pivoted RL agent would do:
  - Use FV's exact gradient (v₂ gap²) as edge scores
  - Sample from softmax(scores/τ) instead of argmax
  - Batch rewire: add swap_frac*m edges, remove swap_frac*m edges per step
  - Re-eigensolve every K step

Variants tested:
  A. greedy:       argmax add, argmin remove (= greedy REFINE baseline)
  B. fixed_temp:   fixed τ for both add and remove, across all K steps
  C. anneal:       τ anneals from τ_start to τ_end over K steps
                   (explore early, exploit late — like LR schedule)
  D. split_temp:   different τ for add (τ_add) vs remove (τ_rem)
                   (maybe add needs more exploration than remove)

Usage:
    python refine_temp_poc.py --n 16 --workers 8
    python refine_temp_poc.py --n 16,24 --workers 8 --k-steps 20
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import linalg
from multiprocessing import Pool, cpu_count

# ---------- Baselines ----------

_cache_path = Path(__file__).parent.parent.parent / "rl" / "baselines_cache.json"
_cache = {}
if _cache_path.exists():
    with open(_cache_path) as f:
        _cache = json.load(f)


def get_fv_baseline(n, m):
    entry = _cache.get(str(n), {}).get(str(m), {})
    return entry.get('fv', 0.0)


def get_best_baseline(n, m):
    entry = _cache.get(str(n), {}).get(str(m), {})
    vals = [v for v in entry.values() if isinstance(v, (int, float)) and v > 0]
    return max(vals) if vals else 0.0


# ---------- Graph utilities ----------

def ring_plus_random(n, m, rng):
    """Ring graph + random edges to reach m total edges."""
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


def fiedler_scores(adj, n):
    """Compute Fiedler gap scores for all (i,j) pairs."""
    L = np.diag(adj.sum(axis=1)) - adj
    evals, evecs = linalg.eigh(L)
    v2 = evecs[:, 1]
    diff = v2[:, None] - v2[None, :]
    return diff ** 2, evals[1]


def sample_from_scores(scores, mask, tau, rng, maximize=True):
    """Sample an edge from masked scores using softmax temperature.

    Args:
        scores: (n, n) score matrix
        mask: (n, n) boolean mask of valid edges
        tau: temperature. 0 = argmax/argmin
        rng: random generator
        maximize: if True, prefer high scores (ADD); if False, prefer low (REMOVE)
    """
    valid_idx = np.where(mask.ravel())[0]
    if len(valid_idx) == 0:
        return None
    valid_scores = scores.ravel()[valid_idx]

    if not maximize:
        valid_scores = -valid_scores  # flip for minimize

    if tau <= 1e-10:
        best = valid_idx[np.argmax(valid_scores)]
    else:
        logits = valid_scores / tau
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        best = valid_idx[rng.choice(len(valid_idx), p=probs)]

    n = scores.shape[0]
    return best // n, best % n


# ---------- Bridge detection ----------

def find_bridges(adj, n):
    """Find bridge edges using DFS. Returns set of (min_i, max_j) tuples."""
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

    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    return bridges


# ---------- REFINE + Temperature ----------

def refine_temp(n, m, k_steps, swap_frac, tau_add, tau_rem, anneal=False,
                tau_add_end=None, tau_rem_end=None, seed=0):
    """
    REFINE-style rewiring with temperature-controlled FV gradient.

    Args:
        tau_add: temperature for ADD phase
        tau_rem: temperature for REMOVE phase
        anneal: if True, linearly anneal tau from start to end over K steps
        tau_add_end/tau_rem_end: end temperatures for annealing
    """
    rng = np.random.default_rng(seed)
    adj = ring_plus_random(n, m, rng)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    num_swaps = max(1, int(m * swap_frac))

    for k in range(k_steps):
        # Compute annealed temperature for this step
        if anneal and tau_add_end is not None:
            progress = k / max(k_steps - 1, 1)
            t_add = tau_add + (tau_add_end - tau_add) * progress
            t_rem = tau_rem + (tau_rem_end - tau_rem) * progress
        else:
            t_add = tau_add
            t_rem = tau_rem

        # Fresh eigendecomposition
        scores, lam2 = fiedler_scores(adj, n)

        # === ADD PHASE: pick non-edges with large Fiedler gap ===
        add_mask = (adj == 0) & upper_tri
        added_edges = []

        for _ in range(num_swaps):
            result = sample_from_scores(scores, add_mask, t_add, rng,
                                        maximize=True)
            if result is None:
                break
            ai, aj = result
            adj[ai, aj] = adj[aj, ai] = 1
            add_mask[ai, aj] = False
            added_edges.append((ai, aj))

        if not added_edges:
            continue

        # === REMOVE PHASE: pick edges with small Fiedler gap ===
        # Recompute scores after additions
        scores, _ = fiedler_scores(adj, n)
        bridges = find_bridges(adj, n)

        rem_mask = (adj > 0) & upper_tri
        # Don't remove bridges
        for bi, bj in bridges:
            rem_mask[bi, bj] = False
        # Don't remove edges we just added
        for ai, aj in added_edges:
            a_min, a_max = min(ai, aj), max(ai, aj)
            rem_mask[a_min, a_max] = False

        removed = 0
        for _ in range(len(added_edges)):
            result = sample_from_scores(scores, rem_mask, t_rem, rng,
                                        maximize=False)  # minimize = remove small gap
            if result is None:
                break
            ri, rj = result
            adj[ri, rj] = adj[rj, ri] = 0
            rem_mask[ri, rj] = False
            removed += 1

        # Undo unmatched adds (maintain edge count)
        unmatched = len(added_edges) - removed
        if unmatched > 0:
            for ai, aj in reversed(added_edges[-unmatched:]):
                if adj[ai, aj] > 0:
                    adj[ai, aj] = adj[aj, ai] = 0

    return algebraic_connectivity(adj)


# ---------- Experimental configurations ----------

CONFIGS = {
    # Baseline: pure greedy (= greedy REFINE)
    'greedy': {
        'tau_add': 0.0, 'tau_rem': 0.0,
        'anneal': False,
    },

    # Fixed small temperature
    'fixed_0.001': {
        'tau_add': 0.001, 'tau_rem': 0.001,
        'anneal': False,
    },
    'fixed_0.005': {
        'tau_add': 0.005, 'tau_rem': 0.005,
        'anneal': False,
    },
    'fixed_0.01': {
        'tau_add': 0.01, 'tau_rem': 0.01,
        'anneal': False,
    },
    'fixed_0.05': {
        'tau_add': 0.05, 'tau_rem': 0.05,
        'anneal': False,
    },

    # Annealing: explore early → exploit late
    'anneal_0.05→0': {
        'tau_add': 0.05, 'tau_rem': 0.05,
        'anneal': True,
        'tau_add_end': 0.0, 'tau_rem_end': 0.0,
    },
    'anneal_0.1→0.001': {
        'tau_add': 0.1, 'tau_rem': 0.1,
        'anneal': True,
        'tau_add_end': 0.001, 'tau_rem_end': 0.001,
    },

    # Split: different temp for add vs remove
    'split_a0.01_r0': {
        'tau_add': 0.01, 'tau_rem': 0.0,
        'anneal': False,
    },
    'split_a0_r0.01': {
        'tau_add': 0.0, 'tau_rem': 0.01,
        'anneal': False,
    },
}

K_STEPS = 20
SWAP_FRAC = 0.05
NUM_SEEDS = 5
SAMPLES_PER_SEED = 3


def _eval_config(args):
    n, m, k_steps, swap_frac = args
    fv_bl = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    results = {'n': n, 'm': m, 'fv_db': fv_bl, 'best_bl': best_bl}

    for name, cfg in CONFIGS.items():
        is_greedy = cfg['tau_add'] <= 1e-10 and cfg['tau_rem'] <= 1e-10 \
                    and not cfg.get('anneal', False)

        if is_greedy:
            # Deterministic — just vary spanning tree seed
            best = max(
                refine_temp(n, m, k_steps, swap_frac,
                            cfg['tau_add'], cfg['tau_rem'],
                            anneal=cfg.get('anneal', False),
                            tau_add_end=cfg.get('tau_add_end'),
                            tau_rem_end=cfg.get('tau_rem_end'),
                            seed=s)
                for s in range(NUM_SEEDS)
            )
        else:
            # Stochastic — more rollouts
            best = max(
                refine_temp(n, m, k_steps, swap_frac,
                            cfg['tau_add'], cfg['tau_rem'],
                            anneal=cfg.get('anneal', False),
                            tau_add_end=cfg.get('tau_add_end'),
                            tau_rem_end=cfg.get('tau_rem_end'),
                            seed=s * 1000 + sample)
                for s in range(NUM_SEEDS)
                for sample in range(SAMPLES_PER_SEED)
            )
        results[name] = best

    return results


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="REFINE + Temperature PoC")
    parser.add_argument("--n", type=str, required=True)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8))
    parser.add_argument("--k-steps", type=int, default=20)
    parser.add_argument("--swap-frac", type=float, default=0.05)
    args = parser.parse_args()

    global K_STEPS, SWAP_FRAC
    K_STEPS = args.k_steps
    SWAP_FRAC = args.swap_frac

    n_values = [int(x.strip()) for x in args.n.split(',')]

    tasks = []
    for n in n_values:
        max_m = n * (n - 1) // 2
        for m in range(n + 1, max_m + 1):
            if get_best_baseline(n, m) > 0:
                tasks.append((n, m, args.k_steps, args.swap_frac))

    config_names = list(CONFIGS.keys())
    print(f"REFINE + Temperature PoC")
    print(f"  n={','.join(str(n) for n in n_values)} | {len(tasks)} configs | "
          f"{args.workers} workers")
    print(f"  K={args.k_steps}, swap_frac={args.swap_frac}")
    print(f"  Seeds={NUM_SEEDS}, Samples/seed={SAMPLES_PER_SEED}")
    print(f"  Greedy: {NUM_SEEDS} rollouts | "
          f"Stochastic: {NUM_SEEDS * SAMPLES_PER_SEED} rollouts")
    print(f"\n  Configs: {', '.join(config_names)}\n")

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue

        max_m = n * (n - 1) // 2
        avg_fv = np.mean([r['fv_db'] for r in nr if r['fv_db'] > 0])
        avg_best = np.mean([r['best_bl'] for r in nr if r['best_bl'] > 0])

        print(f"\n{'='*95}")
        print(f"N = {n} ({len(nr)} configs) | K={args.k_steps}, frac={args.swap_frac}")
        print(f"{'='*95}")
        print(f"  {'Method':<22} {'avg λ₂':>8} {'vs FV_db':>9} "
              f"{'W':>4} {'T':>3} {'vs best':>9} {'W':>4}")
        print(f"  {'-'*66}")

        for name in config_names:
            avg = np.mean([r[name] for r in nr])
            ratio_fv = avg / avg_fv * 100 if avg_fv > 0 else 0
            ratio_best = avg / avg_best * 100 if avg_best > 0 else 0
            wins_fv = sum(1 for r in nr if r[name] > r['fv_db'] + 1e-6
                          and r['fv_db'] > 0)
            ties_fv = sum(1 for r in nr if abs(r[name] - r['fv_db']) < 1e-6
                          and r['fv_db'] > 0)
            wins_best = sum(1 for r in nr if r[name] > r['best_bl'] + 1e-6
                            and r['best_bl'] > 0)
            print(f"  {name:<22} {avg:8.3f} {ratio_fv:8.1f}% "
                  f"{wins_fv:4d} {ties_fv:3d} {ratio_best:8.1f}% {wins_best:4d}")

        # Oracle
        nongreedy_names = [n for n in config_names if n != 'greedy']
        oracle_vals = [max(r[n] for n in config_names) for r in nr]
        oracle_avg = np.mean(oracle_vals)
        oracle_wins = sum(1 for r, ov in zip(nr, oracle_vals)
                          if ov > r['best_bl'] + 1e-6 and r['best_bl'] > 0)
        print(f"\n  Oracle (per-config best): avg={oracle_avg:.3f} | "
              f"vs FV_db: {oracle_avg/avg_fv*100:.1f}% | "
              f"vs best: {oracle_avg/avg_best*100:.1f}% ({oracle_wins}W)")

        # Per-config improvement
        improved = 0
        total_gain = 0.0
        for r in nr:
            best_ng = max(r[n] for n in nongreedy_names)
            if best_ng > r['greedy'] + 1e-6:
                improved += 1
                total_gain += best_ng - r['greedy']
        print(f"  Non-greedy beats greedy: {improved}/{len(nr)} configs "
              f"(+{total_gain:.3f} total λ₂)")

        # Density breakdown
        print(f"\n  By density (best method):")
        density_bins = [
            ("sparse  <0.2", 0.0, 0.2),
            ("medium 0.2-0.5", 0.2, 0.5),
            ("dense  0.5-0.8", 0.5, 0.8),
            ("v.dense ≥0.8", 0.8, 1.01),
        ]
        for label, lo, hi in density_bins:
            subset = [r for r in nr if lo <= r['m'] / max_m < hi]
            if not subset:
                continue
            best_name = None
            best_avg = 0
            for name in config_names:
                avg = np.mean([r[name] for r in subset])
                if avg > best_avg:
                    best_avg = avg
                    best_name = name
            greedy_avg = np.mean([r['greedy'] for r in subset])
            bl_avg = np.mean([r['best_bl'] for r in subset
                              if r['best_bl'] > 0])
            delta = best_avg - greedy_avg
            print(f"    {label}: {best_name:<22} "
                  f"(avg={best_avg:.3f}, greedy={greedy_avg:.3f}, "
                  f"Δ={delta:+.3f}, vs_bl={best_avg/bl_avg*100:.1f}%)")


if __name__ == "__main__":
    main()
