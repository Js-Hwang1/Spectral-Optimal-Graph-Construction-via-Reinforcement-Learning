#!/usr/bin/env python3
"""
PoC: FV construction with cascading eigenvector fallback.

When λ_k ≈ λ_{k+1}, the k-th eigenvector is unstable. This PoC cascades
through eigenvectors 2..10: for each near-degenerate pair, blend in the
higher eigenvector's gap into the scoring.

Variants tested:
  - FV standard:     always v₂ gap
  - cascade_max:     max(v_k gap²) over all near-degenerate k=2..K
  - cascade_weighted: v₂_gap² + Σ w_k * v_k_gap² for near-degenerate k
  - cascade_sum:     Σ v_k_gap² over all near-degenerate k=2..K

Usage:
    python fv_cascade_poc.py --n 16 --workers 8
    python fv_cascade_poc.py --n 16,24 --workers 8 --max-k 10
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
    n_data = _cache.get(str(n), {})
    entry = n_data.get(str(m), {})
    return entry.get('fv', 0.0)


def get_best_baseline(n, m):
    n_data = _cache.get(str(n), {})
    entry = n_data.get(str(m), {})
    vals = [v for v in entry.values() if isinstance(v, (int, float)) and v > 0]
    return max(vals) if vals else 0.0


# ---------- Graph utilities ----------

def random_spanning_tree(n, rng):
    """Random spanning tree via Wilson's algorithm."""
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


def laplacian(adj):
    return np.diag(adj.sum(axis=1)) - adj


def algebraic_connectivity(adj):
    evals = linalg.eigvalsh(laplacian(adj))
    return float(evals[1])


def eigen_decomp(adj, max_k=10):
    """Return eigenvalues and eigenvectors of Laplacian (first max_k+1)."""
    L = laplacian(adj)
    evals, evecs = linalg.eigh(L)
    return evals[:max_k + 1], evecs[:, :max_k + 1]


# ---------- Scoring functions ----------

def score_standard(adj, evals, evecs, n, max_k, threshold):
    """Standard FV: v₂ gap only."""
    v2 = evecs[:, 1]
    return (v2[:, None] - v2[None, :]) ** 2


def _contiguous_block(evals, threshold):
    """Find contiguous degenerate block starting from λ₂.
    Returns the index of the last eigenvalue in the block (1-indexed)."""
    last = 1  # at minimum, we use v₂ (index 1)
    for k in range(1, len(evals) - 1):
        lam_k = evals[k]
        lam_next = evals[k + 1]
        if lam_k < 1e-8:
            continue
        gap = (lam_next - lam_k) / lam_k
        if gap < threshold:
            last = k + 1
        else:
            break  # first real gap — stop
    return last


def score_cascade_max(adj, evals, evecs, n, max_k, threshold):
    """Max gap over contiguous degenerate block from λ₂."""
    last = _contiguous_block(evals, threshold)
    v2 = evecs[:, 1]
    scores = (v2[:, None] - v2[None, :]) ** 2

    for k in range(2, last + 1):
        vk = evecs[:, k]
        diff_k = (vk[:, None] - vk[None, :]) ** 2
        scores = np.maximum(scores, diff_k)

    return scores


def score_cascade_weighted(adj, evals, evecs, n, max_k, threshold):
    """Weighted sum over contiguous degenerate block from λ₂."""
    last = _contiguous_block(evals, threshold)
    v2 = evecs[:, 1]
    scores = (v2[:, None] - v2[None, :]) ** 2

    for k in range(1, last):
        lam_k = evals[k]
        lam_next = evals[k + 1]
        if lam_k < 1e-8:
            continue
        gap = (lam_next - lam_k) / lam_k
        w = 1.0 - gap / threshold
        vk1 = evecs[:, k + 1]
        diff_k1 = (vk1[:, None] - vk1[None, :]) ** 2
        scores = scores + w * diff_k1

    return scores


def score_cascade_sum(adj, evals, evecs, n, max_k, threshold):
    """Equal-weight sum over contiguous degenerate block from λ₂."""
    last = _contiguous_block(evals, threshold)
    scores = np.zeros((n, n))

    for k in range(1, last + 1):
        vk = evecs[:, k]
        scores += (vk[:, None] - vk[None, :]) ** 2

    return scores


# ---------- Generic FV construction ----------

SCORE_FNS = {
    'standard': score_standard,
    'cascade_max': score_cascade_max,
    'cascade_weighted': score_cascade_weighted,
    'cascade_sum': score_cascade_sum,
}


def fv_construct(n, m, score_fn, max_k=10, threshold=0.3, seed=0):
    """Construct graph using FV with given scoring function."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)

    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    for _ in range(edges_to_add):
        evals, evecs = eigen_decomp(adj, max_k)
        scores = score_fn(adj, evals, evecs, n, max_k, threshold)

        mask = (adj == 0) & upper_tri
        if not mask.any():
            break

        scores_masked = np.where(mask, scores, -1)
        idx = np.unravel_index(np.argmax(scores_masked), scores.shape)
        i, j = idx
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


# ---------- Worker ----------

def _eval_config(args):
    n, m, max_k, threshold, num_seeds = args
    fv_bl = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    results = {'n': n, 'm': m, 'fv_db': fv_bl, 'best_bl': best_bl}

    for name, fn in SCORE_FNS.items():
        best = max(
            fv_construct(n, m, fn, max_k=max_k, threshold=threshold, seed=s)
            for s in range(num_seeds)
        )
        results[name] = best

    return results


# ---------- Main ----------

def main():
    global MAX_K, THRESHOLD, NUM_SEEDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=str, required=True)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8))
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    MAX_K = args.max_k
    THRESHOLD = args.threshold
    NUM_SEEDS = args.seeds

    n_values = [int(x.strip()) for x in args.n.split(',')]

    max_k = args.max_k
    threshold = args.threshold
    num_seeds = args.seeds

    tasks = []
    for n in n_values:
        max_m = n * (n - 1) // 2
        for m in range(n + 1, max_m + 1):
            if get_best_baseline(n, m) > 0:
                tasks.append((n, m, max_k, threshold, num_seeds))

    print(f"FV Cascade PoC | n={args.n} | {len(tasks)} configs | "
          f"{args.workers} workers")
    print(f"max_k={MAX_K}, threshold={THRESHOLD}, seeds={NUM_SEEDS}\n")

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    methods = list(SCORE_FNS.keys())

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue

        print(f"\n{'='*90}")
        print(f"N = {n} ({len(nr)} configs) | max_k={MAX_K}, threshold={THRESHOLD}")
        print(f"{'='*90}")

        for name in methods:
            avg = np.mean([r[name] for r in nr])
            avg_fv = np.mean([r['fv_db'] for r in nr if r['fv_db'] > 0])
            avg_best = np.mean([r['best_bl'] for r in nr if r['best_bl'] > 0])
            wins_fv = sum(1 for r in nr if r[name] > r['fv_db'] + 1e-6
                          and r['fv_db'] > 0)
            ties_fv = sum(1 for r in nr if abs(r[name] - r['fv_db']) < 1e-6
                          and r['fv_db'] > 0)
            wins_best = sum(1 for r in nr if r[name] > r['best_bl'] + 1e-6
                            and r['best_bl'] > 0)
            ratio_fv = avg / avg_fv * 100 if avg_fv > 0 else 0
            ratio_best = avg / avg_best * 100 if avg_best > 0 else 0
            print(f"  {name:22s}: avg={avg:8.3f} | vs FV_db: {ratio_fv:5.1f}% "
                  f"({wins_fv:3d}W {ties_fv:2d}T) | "
                  f"vs best: {ratio_best:5.1f}% ({wins_best:3d}W)")

        # Best cascade method vs standard
        print(f"\n  Per-config best cascade vs standard:")
        cascade_methods = [m for m in methods if m != 'standard']
        improved = 0
        hurt = 0
        total_gain = 0.0
        for r in nr:
            best_cascade = max(r[m] for m in cascade_methods)
            if best_cascade > r['standard'] + 1e-6:
                improved += 1
                total_gain += best_cascade - r['standard']
            elif best_cascade < r['standard'] - 1e-6:
                hurt += 1

        print(f"    Improved: {improved}/{len(nr)} configs "
              f"(+{total_gain:.3f} total gain)")
        print(f"    Hurt: {hurt}/{len(nr)} configs")


if __name__ == "__main__":
    main()
