#!/usr/bin/env python3
"""
PoC: Oracle Rank Analysis — Where is the true best edge in FV's ordering?

=== Question ===
When FV picks the wrong edge, is the RIGHT edge nearby in FV's ranking
(directional — small perturbation suffices) or far away (probabilistic —
need random search)?

=== Method ===
At each step of FV construction, brute-force compute exact Δλ₂ for ALL
candidate non-edges (feasible at n≤24). This gives us the "oracle" — the
truly optimal per-step choice.

We then measure:
  1. FV accuracy: how often does FV pick the oracle's choice?
  2. Oracle rank in FV ordering: when FV is wrong, how far off is it?
  3. Top-K capture: if we evaluate FV's top-K and pick the true best, how
     close to oracle do we get?
  4. Full rollout: build graphs with FV, oracle, and top-K strategies.

=== Interpretation ===
  Oracle consistently in FV top-3 → DIRECTIONAL (FV's gradient is right
    direction, just needs refinement among top candidates)
  Oracle scattered (rank 10+) → PROBABILISTIC (FV's gradient is misleading,
    need random exploration to escape)

Usage:
    python oracle_rank_poc.py --n 16 --workers 8
    python oracle_rank_poc.py --n 16,24 --workers 8
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import linalg
from multiprocessing import Pool, cpu_count
from collections import defaultdict

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

def random_spanning_tree(n, rng):
    adj = np.zeros((n, n), dtype=np.float64)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    for start in range(1, n):
        if in_tree[start]:
            continue
        nxt = np.zeros(n, dtype=int)
        cur = start
        while not in_tree[cur]:
            nxt[cur] = rng.integers(0, n)
            while nxt[cur] == cur:
                nxt[cur] = rng.integers(0, n)
            cur = nxt[cur]
        cur = start
        while not in_tree[cur]:
            in_tree[cur] = True
            adj[cur, nxt[cur]] = adj[nxt[cur], cur] = 1
            cur = nxt[cur]
    return adj


def algebraic_connectivity(adj):
    L = np.diag(adj.sum(axis=1)) - adj
    return float(linalg.eigvalsh(L)[1])


# ===== Brute-force oracle =====

def brute_force_deltas(adj, n, lam2, valid_idx):
    """Exact Δλ₂ for every candidate edge. O(N² × N³) per step."""
    deltas = np.zeros(len(valid_idx))
    for k, flat in enumerate(valid_idx):
        i, j = flat // n, flat % n
        adj[i, j] = adj[j, i] = 1
        deltas[k] = float(linalg.eigvalsh(
            np.diag(adj.sum(axis=1)) - adj)[1]) - lam2
        adj[i, j] = adj[j, i] = 0  # undo (in-place, no copy)
    return deltas


# ===== Per-step analysis (follows FV, records oracle info) =====

def analyze_fv_steps(n, m, seed):
    """Run FV construction, record oracle info at each step."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    if edges_to_add <= 0:
        return []

    max_m = n * (n - 1) // 2
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    steps = []

    for step in range(edges_to_add):
        L = np.diag(adj.sum(axis=1)) - adj
        evals, evecs = linalg.eigh(L)
        lam2 = evals[1]
        v2 = evecs[:, 1]

        mask = (adj == 0) & upper_tri
        if not mask.any():
            break
        valid_idx = np.where(mask.ravel())[0]
        num_cand = len(valid_idx)

        # FV scores
        fv_scores = np.array([
            (v2[idx // n] - v2[idx % n]) ** 2 for idx in valid_idx])

        # Oracle: exact Δλ₂ for all candidates
        oracle_deltas = brute_force_deltas(adj, n, lam2, valid_idx)

        fv_choice = np.argmax(fv_scores)
        oracle_choice = np.argmax(oracle_deltas)

        # Rank of oracle's best in FV's descending order
        oracle_rank = int(np.sum(fv_scores > fv_scores[oracle_choice]))

        # Top-K capture: best oracle Δλ₂ among FV's top-K
        fv_order = np.argsort(-fv_scores)  # descending
        topk_deltas = {}
        for K in [1, 3, 5, 10, 20]:
            top_k = fv_order[:min(K, num_cand)]
            topk_deltas[K] = float(np.max(oracle_deltas[top_k]))

        gap_ratio = (evals[2] - lam2) / max(lam2, 1e-8) if lam2 > 1e-8 else 999
        density = (n - 1 + step) / max_m

        steps.append({
            'rank': oracle_rank,
            'num_cand': num_cand,
            'fv_delta': float(oracle_deltas[fv_choice]),
            'oracle_delta': float(oracle_deltas[oracle_choice]),
            'topk': topk_deltas,
            'gap_ratio': float(gap_ratio),
            'density': density,
            'step_frac': step / max(edges_to_add - 1, 1),
            'exact_match': int(fv_choice == oracle_choice),
        })

        # Follow FV's choice
        chosen = valid_idx[fv_choice]
        i, j = chosen // n, chosen % n
        adj[i, j] = adj[j, i] = 1

    return steps


# ===== Full rollout builders =====

def build_fv(n, m, seed):
    """Standard FV greedy construction."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    for _ in range(m - (n - 1)):
        L = np.diag(adj.sum(axis=1)) - adj
        _, evecs = linalg.eigh(L)
        v2 = evecs[:, 1]
        scores = (v2[:, None] - v2[None, :]) ** 2
        mask = (adj == 0) & upper_tri
        if not mask.any():
            break
        scores[~mask] = -1
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        adj[idx[0], idx[1]] = adj[idx[1], idx[0]] = 1
    return algebraic_connectivity(adj)


def build_oracle(n, m, seed):
    """Oracle: always pick edge with highest TRUE Δλ₂."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    for _ in range(m - (n - 1)):
        L = np.diag(adj.sum(axis=1)) - adj
        lam2 = float(linalg.eigvalsh(L)[1])
        mask = (adj == 0) & upper_tri
        if not mask.any():
            break
        valid_idx = np.where(mask.ravel())[0]
        deltas = brute_force_deltas(adj, n, lam2, valid_idx)
        best = valid_idx[np.argmax(deltas)]
        i, j = best // n, best % n
        adj[i, j] = adj[j, i] = 1
    return algebraic_connectivity(adj)


def build_topk(n, m, K, seed):
    """Top-K: pick best TRUE Δλ₂ among FV's top-K candidates."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    for _ in range(m - (n - 1)):
        L = np.diag(adj.sum(axis=1)) - adj
        evals, evecs = linalg.eigh(L)
        lam2 = evals[1]
        v2 = evecs[:, 1]
        mask = (adj == 0) & upper_tri
        if not mask.any():
            break
        valid_idx = np.where(mask.ravel())[0]
        fv_scores = np.array([
            (v2[idx // n] - v2[idx % n]) ** 2 for idx in valid_idx])
        # FV's top-K
        top_k_local = np.argsort(-fv_scores)[:min(K, len(valid_idx))]
        top_k_global = valid_idx[top_k_local]
        # Evaluate true Δλ₂ for just these K
        best_delta = -np.inf
        best_edge = top_k_global[0]
        for flat in top_k_global:
            i, j = flat // n, flat % n
            adj[i, j] = adj[j, i] = 1
            d = float(linalg.eigvalsh(
                np.diag(adj.sum(axis=1)) - adj)[1]) - lam2
            adj[i, j] = adj[j, i] = 0
            if d > best_delta:
                best_delta = d
                best_edge = flat
        i, j = best_edge // n, best_edge % n
        adj[i, j] = adj[j, i] = 1
    return algebraic_connectivity(adj)


# ===== Worker =====

NUM_SEEDS = 3


def _eval_config(args):
    n, m = args
    fv_db = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    # Step analysis (1 seed, plenty of steps for statistics)
    step_data = analyze_fv_steps(n, m, seed=0)

    # Rollouts (best of NUM_SEEDS)
    fv_best = max(build_fv(n, m, s) for s in range(NUM_SEEDS))
    oracle_best = max(build_oracle(n, m, s) for s in range(NUM_SEEDS))
    top3_best = max(build_topk(n, m, 3, s) for s in range(NUM_SEEDS))
    top5_best = max(build_topk(n, m, 5, s) for s in range(NUM_SEEDS))
    top10_best = max(build_topk(n, m, 10, s) for s in range(NUM_SEEDS))

    return {
        'n': n, 'm': m,
        'fv_db': fv_db, 'best_bl': best_bl,
        'fv': fv_best, 'oracle': oracle_best,
        'top3': top3_best, 'top5': top5_best, 'top10': top10_best,
        'steps': step_data,
    }


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description="Oracle Rank Analysis: where is the true best edge?")
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

    print(f"Oracle Rank PoC | n={args.n} | {len(tasks)} configs | "
          f"{args.workers} workers | {NUM_SEEDS} seeds")
    print()

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        max_m = n * (n - 1) // 2

        # Collect all steps
        all_steps = []
        for r in nr:
            all_steps.extend(r['steps'])

        total_steps = len(all_steps)

        print(f"\n{'=' * 90}")
        print(f"N = {n} ({len(nr)} configs, {total_steps} total steps)")
        print(f"{'=' * 90}")

        # --- 1. FV accuracy ---
        exact = sum(s['exact_match'] for s in all_steps)
        print(f"\n  1. FV accuracy (picks oracle's #1 choice):")
        print(f"     {exact}/{total_steps} = {exact / total_steps * 100:.1f}%")

        # --- 2. Rank distribution ---
        ranks = [s['rank'] for s in all_steps]
        print(f"\n  2. Oracle's rank in FV ordering (0 = FV's top pick):")
        print(f"     Median: {np.median(ranks):.0f}")
        print(f"     Mean:   {np.mean(ranks):.1f}")
        print(f"     P90:    {np.percentile(ranks, 90):.0f}")
        print(f"     P99:    {np.percentile(ranks, 99):.0f}")

        for K in [1, 3, 5, 10, 20]:
            frac = sum(1 for r in ranks if r < K) / total_steps
            print(f"     In top-{K:2d}: {frac * 100:5.1f}%")

        # --- 3. Per-step Δλ₂ capture ---
        print(f"\n  3. Per-step Δλ₂ capture (FV's choice vs oracle):")
        fv_deltas = [s['fv_delta'] for s in all_steps]
        oracle_deltas = [s['oracle_delta'] for s in all_steps]
        avg_fv = np.mean(fv_deltas)
        avg_oracle = np.mean(oracle_deltas)
        print(f"     FV avg Δλ₂:     {avg_fv:.6f}")
        print(f"     Oracle avg Δλ₂: {avg_oracle:.6f}")
        print(f"     Capture ratio:  {avg_fv / avg_oracle * 100:.1f}%")
        print(f"     Avg regret:     {avg_oracle - avg_fv:.6f}")

        # Top-K per-step capture
        print(f"\n     Top-K per-step Δλ₂ capture:")
        for K in [1, 3, 5, 10, 20]:
            avg_topk = np.mean([s['topk'][K] for s in all_steps])
            print(f"       top-{K:2d}: avg Δλ₂={avg_topk:.6f} "
                  f"({avg_topk / avg_oracle * 100:.1f}% of oracle)")

        # --- 4. Breakdown by gap ratio ---
        print(f"\n  4. Breakdown by spectral gap (λ₃-λ₂)/λ₂:")
        gap_bins = [
            ("degen <0.1", 0, 0.1),
            ("near  0.1-0.3", 0.1, 0.3),
            ("clear 0.3-1.0", 0.3, 1.0),
            ("wide  ≥1.0", 1.0, 1000),
        ]
        for label, lo, hi in gap_bins:
            sub = [s for s in all_steps if lo <= s['gap_ratio'] < hi]
            if not sub:
                continue
            acc = sum(s['exact_match'] for s in sub) / len(sub)
            med_rank = np.median([s['rank'] for s in sub])
            cap = np.mean([s['fv_delta'] for s in sub]) / max(
                np.mean([s['oracle_delta'] for s in sub]), 1e-12)
            top3_cap = np.mean([s['topk'][3] for s in sub]) / max(
                np.mean([s['oracle_delta'] for s in sub]), 1e-12)
            print(f"     {label}: {len(sub):4d} steps | "
                  f"FV acc={acc * 100:5.1f}% | "
                  f"med rank={med_rank:4.0f} | "
                  f"FV cap={cap * 100:5.1f}% | "
                  f"top3 cap={top3_cap * 100:5.1f}%")

        # --- 5. Breakdown by density ---
        print(f"\n  5. Breakdown by density:")
        dens_bins = [
            ("sparse  <0.3", 0.0, 0.3),
            ("medium 0.3-0.6", 0.3, 0.6),
            ("dense   ≥0.6", 0.6, 1.01),
        ]
        for label, lo, hi in dens_bins:
            sub = [s for s in all_steps if lo <= s['density'] < hi]
            if not sub:
                continue
            acc = sum(s['exact_match'] for s in sub) / len(sub)
            med_rank = np.median([s['rank'] for s in sub])
            print(f"     {label}: {len(sub):4d} steps | "
                  f"FV acc={acc * 100:5.1f}% | "
                  f"med rank={med_rank:4.0f}")

        # --- 6. Full rollout comparison ---
        print(f"\n  6. Full rollout (best-of-{NUM_SEEDS} seeds):")
        avg_fv_db = np.mean([r['fv_db'] for r in nr])
        avg_best = np.mean([r['best_bl'] for r in nr])

        methods = [
            ('FV (database)', 'fv_db'),
            ('FV (span tree)', 'fv'),
            ('Top-3 + oracle', 'top3'),
            ('Top-5 + oracle', 'top5'),
            ('Top-10 + oracle', 'top10'),
            ('Full oracle', 'oracle'),
            ('Best baseline', 'best_bl'),
        ]
        print(f"     {'Method':<20} {'avg λ₂':>8} {'vs FV_db':>9} "
              f"{'vs best':>9}")
        print(f"     {'-' * 50}")
        for label, key in methods:
            avg = np.mean([r[key] for r in nr])
            r_fv = avg / avg_fv_db * 100 if avg_fv_db > 0 else 0
            r_best = avg / avg_best * 100 if avg_best > 0 else 0
            print(f"     {label:<20} {avg:8.3f} {r_fv:8.1f}% {r_best:8.1f}%")

        # Oracle improvement over FV
        oracle_wins = sum(1 for r in nr
                          if r['oracle'] > r['fv'] + 1e-6)
        oracle_avg = np.mean([r['oracle'] for r in nr])
        fv_avg = np.mean([r['fv'] for r in nr])
        print(f"\n     Oracle beats FV_ST: {oracle_wins}/{len(nr)} configs")
        print(f"     Oracle avg gain: +{oracle_avg - fv_avg:.3f} λ₂")

        # Top-K wins over FV
        for K, key in [(3, 'top3'), (5, 'top5'), (10, 'top10')]:
            wins = sum(1 for r in nr if r[key] > r['fv'] + 1e-6)
            avg_k = np.mean([r[key] for r in nr])
            gap_closed = ((avg_k - fv_avg) /
                          max(oracle_avg - fv_avg, 1e-8) * 100)
            print(f"     Top-{K} beats FV_ST: {wins}/{len(nr)} | "
                  f"closes {gap_closed:.0f}% of FV→oracle gap")


if __name__ == "__main__":
    main()
