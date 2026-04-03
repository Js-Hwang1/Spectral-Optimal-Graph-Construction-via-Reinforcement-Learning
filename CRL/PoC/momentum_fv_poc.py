#!/usr/bin/env python3
"""
PoC: Momentum FV — Exponential Moving Average of Fiedler vectors.

Hypothesis: FV's greedy gradient ascent fails because v₂ oscillates wildly
when λ₂ ≈ λ₃ (93%+ of steps). By smoothing v₂ across steps with an EMA,
we maintain a consistent "structural direction" and avoid chasing rotations.

Method:
  Standard FV:   score(i,j) = (v₂[i] - v₂[j])²     (recomputed each step)
  Momentum FV:   score(i,j) = (v̄₂[i] - v̄₂[j])²     (v̄₂ = EMA of v₂)

EMA update: v̄₂ ← α * v̄₂ + (1-α) * sign_align(v₂)
  - sign_align corrects for eigenvector sign flips (v₂ and -v₂ are equivalent)
  - α ∈ [0, 1]: α=0 is standard FV, α→1 is pure momentum (ignores new info)

Tested α values: 0.0 (baseline), 0.3, 0.5, 0.7, 0.9
Initialization: Random spanning tree, 5 seeds, best-of-5.

Usage:
    python momentum_fv_poc.py --n 16 --workers 8
    python momentum_fv_poc.py --n 16,24 --workers 8
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


# ---------- Momentum FV ----------

def momentum_fv(n, m, alpha, seed=0):
    """
    FV construction with EMA-smoothed Fiedler vector.

    Args:
        alpha: EMA coefficient. 0.0 = standard FV, higher = more momentum.
    """
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    v2_ema = None  # initialized on first step

    for _ in range(edges_to_add):
        L = np.diag(adj.sum(axis=1)) - adj
        evals, evecs = linalg.eigh(L)
        v2 = evecs[:, 1]

        if v2_ema is None:
            v2_ema = v2.copy()
        else:
            # Sign-align: eigenvectors have arbitrary sign
            if np.dot(v2, v2_ema) < 0:
                v2 = -v2
            # EMA update
            v2_ema = alpha * v2_ema + (1 - alpha) * v2
            # Normalize to unit length (preserve scale-invariance)
            norm = np.linalg.norm(v2_ema)
            if norm > 1e-12:
                v2_ema = v2_ema / norm

        # Score with smoothed vector
        diff = v2_ema[:, None] - v2_ema[None, :]
        scores = diff ** 2

        mask = (adj == 0) & upper_tri
        if not mask.any():
            break

        scores[~mask] = -1
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        adj[idx[0], idx[1]] = adj[idx[1], idx[0]] = 1

    return algebraic_connectivity(adj)


# ---------- Worker ----------

ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9]
NUM_SEEDS = 5


def _eval_config(args):
    n, m = args
    fv_bl = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    results = {'n': n, 'm': m, 'fv_db': fv_bl, 'best_bl': best_bl}

    for alpha in ALPHAS:
        best = max(
            momentum_fv(n, m, alpha=alpha, seed=s)
            for s in range(NUM_SEEDS)
        )
        results[f'a{alpha:.1f}'] = best

    return results


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Momentum FV PoC: EMA-smoothed Fiedler vector")
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

    print(f"Momentum FV PoC | n={','.join(str(n) for n in n_values)} | "
          f"{len(tasks)} configs | {args.workers} workers")
    print(f"Alphas: {ALPHAS} | Seeds: {NUM_SEEDS}")
    print(f"(α=0.0 is standard FV baseline)\n")

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    alpha_keys = [f'a{a:.1f}' for a in ALPHAS]

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue

        max_m = n * (n - 1) // 2
        avg_fv = np.mean([r['fv_db'] for r in nr if r['fv_db'] > 0])
        avg_best = np.mean([r['best_bl'] for r in nr if r['best_bl'] > 0])

        print(f"\n{'='*90}")
        print(f"N = {n} ({len(nr)} configs)")
        print(f"{'='*90}")
        print(f"  {'Method':<18} {'avg λ₂':>8} {'vs FV_db':>9} "
              f"{'W':>4} {'T':>3} {'vs best':>9} {'W':>4}")
        print(f"  {'-'*60}")

        for ak, alpha in zip(alpha_keys, ALPHAS):
            avg = np.mean([r[ak] for r in nr])
            ratio_fv = avg / avg_fv * 100 if avg_fv > 0 else 0
            ratio_best = avg / avg_best * 100 if avg_best > 0 else 0
            wins_fv = sum(1 for r in nr if r[ak] > r['fv_db'] + 1e-6
                          and r['fv_db'] > 0)
            ties_fv = sum(1 for r in nr if abs(r[ak] - r['fv_db']) < 1e-6
                          and r['fv_db'] > 0)
            wins_best = sum(1 for r in nr if r[ak] > r['best_bl'] + 1e-6
                            and r['best_bl'] > 0)
            label = f"α={alpha:.1f}" + (" (std FV)" if alpha == 0.0 else "")
            print(f"  {label:<18} {avg:8.3f} {ratio_fv:8.1f}% "
                  f"{wins_fv:4d} {ties_fv:3d} {ratio_best:8.1f}% {wins_best:4d}")

        # Per-config: best momentum α vs α=0
        print(f"\n  Per-config best momentum (α>0) vs standard (α=0):")
        momentum_keys = [k for k in alpha_keys if k != 'a0.0']
        improved = 0
        total_gain = 0.0
        for r in nr:
            best_mom = max(r[k] for k in momentum_keys)
            if best_mom > r['a0.0'] + 1e-6:
                improved += 1
                total_gain += best_mom - r['a0.0']
        print(f"    Improved: {improved}/{len(nr)} configs "
              f"(+{total_gain:.3f} total λ₂ gain)")

        # Breakdown by density
        print(f"\n  By density:")
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
            # Find best alpha for this density bin
            best_alpha = None
            best_avg = 0
            for ak, alpha in zip(alpha_keys, ALPHAS):
                avg = np.mean([r[ak] for r in subset])
                if avg > best_avg:
                    best_avg = avg
                    best_alpha = alpha
            std_avg = np.mean([r['a0.0'] for r in subset])
            bl_avg = np.mean([r['best_bl'] for r in subset
                              if r['best_bl'] > 0])
            delta = best_avg - std_avg
            print(f"    {label}: best α={best_alpha:.1f} "
                  f"(avg={best_avg:.3f}, std={std_avg:.3f}, "
                  f"Δ={delta:+.3f}, vs_bl={best_avg/bl_avg*100:.1f}%)")


if __name__ == "__main__":
    main()
