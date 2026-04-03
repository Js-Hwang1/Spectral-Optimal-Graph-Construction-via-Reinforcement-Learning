#!/usr/bin/env python3
"""
PoC: FV construction with v₃ fallback when λ₂ ≈ λ₃.

Standard FV greedily adds the non-edge with largest |v₂[i]-v₂[j]|².
When λ₂ ≈ λ₃, v₂ is unstable — small perturbations rotate the eigenspace.
This PoC switches to v₃ gap (or combined v₂+v₃) when the spectral gap is small.

Usage:
    python fv_v3_poc.py --n 16 --trials 3
    python fv_v3_poc.py --n 16,24,32 --trials 3
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

def ring_graph(n):
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1
    return adj


def random_spanning_tree(n, rng=None):
    """Random spanning tree via random walk (Wilson's algorithm)."""
    if rng is None:
        rng = np.random.default_rng()
    adj = np.zeros((n, n), dtype=np.float64)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    nodes_in = 1

    # Random walk from each node until hitting the tree
    for start in range(1, n):
        if in_tree[start]:
            continue
        # Random walk, recording next pointers
        nxt = np.zeros(n, dtype=int)
        cur = start
        while not in_tree[cur]:
            nxt[cur] = rng.integers(0, n)
            while nxt[cur] == cur:
                nxt[cur] = rng.integers(0, n)
            cur = nxt[cur]
        # Trace path back, adding edges
        cur = start
        while not in_tree[cur]:
            in_tree[cur] = True
            adj[cur, nxt[cur]] = adj[nxt[cur], cur] = 1
            cur = nxt[cur]
            nodes_in += 1

    return adj


def laplacian(adj):
    D = np.diag(adj.sum(axis=1))
    return D - adj


def algebraic_connectivity(adj):
    L = laplacian(adj)
    evals = linalg.eigvalsh(L)
    return float(evals[1])


def eigen_decomp(adj):
    """Return eigenvalues and eigenvectors of Laplacian."""
    L = laplacian(adj)
    evals, evecs = linalg.eigh(L)
    return evals, evecs


# ---------- FV construction algorithms ----------

def fv_standard(n, m, seed=0):
    """Standard FV: always pick non-edge with largest |v₂[i]-v₂[j]|²."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)  # spanning tree has n-1 edges

    for _ in range(edges_to_add):
        evals, evecs = eigen_decomp(adj)
        v2 = evecs[:, 1]
        diff = v2[:, None] - v2[None, :]
        scores = diff ** 2

        # Mask existing edges and lower triangle
        mask = (adj == 0)
        np.fill_diagonal(mask, False)
        mask = np.triu(mask)

        if not mask.any():
            break

        scores_masked = np.where(mask, scores, -1)
        idx = np.unravel_index(np.argmax(scores_masked), scores.shape)
        i, j = idx
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


def fv_v3_switch(n, m, threshold=0.3, seed=0):
    """FV with v₃ fallback: when (λ₃-λ₂)/λ₂ < threshold, use v₃ gap."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)

    for _ in range(edges_to_add):
        evals, evecs = eigen_decomp(adj)
        v2 = evecs[:, 1]
        v3 = evecs[:, 2]
        lam2, lam3 = evals[1], evals[2]

        # Check spectral gap
        gap_ratio = (lam3 - lam2) / max(lam2, 1e-10)

        if gap_ratio < threshold and lam2 > 1e-8:
            # Near-degenerate: use v₃ gap instead
            diff = v3[:, None] - v3[None, :]
            scores = diff ** 2
        else:
            # Standard: use v₂ gap
            diff = v2[:, None] - v2[None, :]
            scores = diff ** 2

        mask = (adj == 0)
        np.fill_diagonal(mask, False)
        mask = np.triu(mask)

        if not mask.any():
            break

        scores_masked = np.where(mask, scores, -1)
        idx = np.unravel_index(np.argmax(scores_masked), scores.shape)
        i, j = idx
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


def fv_v3_combined(n, m, threshold=0.3, seed=0):
    """FV with combined score: when λ₂≈λ₃, use max(v₂_gap², v₃_gap²)."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)

    for _ in range(edges_to_add):
        evals, evecs = eigen_decomp(adj)
        v2 = evecs[:, 1]
        v3 = evecs[:, 2]
        lam2, lam3 = evals[1], evals[2]

        diff2 = (v2[:, None] - v2[None, :]) ** 2
        gap_ratio = (lam3 - lam2) / max(lam2, 1e-10)

        if gap_ratio < threshold and lam2 > 1e-8:
            # Near-degenerate: use max of v₂ and v₃ gaps
            diff3 = (v3[:, None] - v3[None, :]) ** 2
            scores = np.maximum(diff2, diff3)
        else:
            scores = diff2

        mask = (adj == 0)
        np.fill_diagonal(mask, False)
        mask = np.triu(mask)

        if not mask.any():
            break

        scores_masked = np.where(mask, scores, -1)
        idx = np.unravel_index(np.argmax(scores_masked), scores.shape)
        i, j = idx
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


def fv_v3_weighted(n, m, threshold=0.3, seed=0):
    """FV with weighted combo: score = v₂_gap² + w * v₃_gap² where w depends on gap."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)

    for _ in range(edges_to_add):
        evals, evecs = eigen_decomp(adj)
        v2 = evecs[:, 1]
        v3 = evecs[:, 2]
        lam2, lam3 = evals[1], evals[2]

        diff2 = (v2[:, None] - v2[None, :]) ** 2
        gap_ratio = (lam3 - lam2) / max(lam2, 1e-10)

        if gap_ratio < threshold and lam2 > 1e-8:
            diff3 = (v3[:, None] - v3[None, :]) ** 2
            # Weight v₃ more as gap shrinks
            w = 1.0 - gap_ratio / threshold  # 1.0 at gap=0, 0.0 at gap=threshold
            scores = diff2 + w * diff3
        else:
            scores = diff2

        mask = (adj == 0)
        np.fill_diagonal(mask, False)
        mask = np.triu(mask)

        if not mask.any():
            break

        scores_masked = np.where(mask, scores, -1)
        idx = np.unravel_index(np.argmax(scores_masked), scores.shape)
        i, j = idx
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


# ---------- Worker ----------

NUM_SEEDS = 5


def _eval_config(args):
    n, m = args
    fv_bl = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    # Run multiple random spanning tree inits, take best
    std_best = max(fv_standard(n, m, seed=s) for s in range(NUM_SEEDS))
    switch_best = max(fv_v3_switch(n, m, seed=s) for s in range(NUM_SEEDS))
    combined_best = max(fv_v3_combined(n, m, seed=s) for s in range(NUM_SEEDS))
    weighted_best = max(fv_v3_weighted(n, m, seed=s) for s in range(NUM_SEEDS))

    return {
        'n': n, 'm': m,
        'fv_db': fv_bl,
        'best_bl': best_bl,
        'std': std_best,
        'switch': switch_best,
        'combined': combined_best,
        'weighted': weighted_best,
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=str, required=True)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8))
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]

    tasks = []
    for n in n_values:
        max_m = n * (n - 1) // 2
        for m in range(n + 1, max_m + 1):  # skip m=n (ring itself)
            if get_best_baseline(n, m) > 0:
                tasks.append((n, m))

    print(f"FV + v₃ PoC | n={args.n} | {len(tasks)} configs | {args.workers} workers")
    print(f"Threshold=0.3 for λ₂≈λ₃ detection\n")

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    # Aggregate by n
    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue

        print(f"\n{'='*80}")
        print(f"N = {n} ({len(nr)} configs)")
        print(f"{'='*80}")

        methods = [
            ('FV (standard)', 'std'),
            ('FV+v₃ switch', 'switch'),
            ('FV+v₃ combined', 'combined'),
            ('FV+v₃ weighted', 'weighted'),
        ]

        for label, key in methods:
            avg = np.mean([r[key] for r in nr])
            avg_fv = np.mean([r['fv_db'] for r in nr if r['fv_db'] > 0])
            avg_best = np.mean([r['best_bl'] for r in nr if r['best_bl'] > 0])
            wins_fv = sum(1 for r in nr if r[key] > r['fv_db'] + 1e-6 and r['fv_db'] > 0)
            ties_fv = sum(1 for r in nr if abs(r[key] - r['fv_db']) < 1e-6 and r['fv_db'] > 0)
            wins_best = sum(1 for r in nr if r[key] > r['best_bl'] + 1e-6 and r['best_bl'] > 0)
            ratio_fv = avg / avg_fv * 100 if avg_fv > 0 else 0
            ratio_best = avg / avg_best * 100 if avg_best > 0 else 0
            print(f"  {label:20s}: avg={avg:.3f} | vs FV_db: {ratio_fv:.1f}% "
                  f"({wins_fv}W {ties_fv}T) | vs best: {ratio_best:.1f}% ({wins_best}W)")

        # Show configs where v₃ methods beat standard FV
        improvements = []
        for r in nr:
            best_v3 = max(r['switch'], r['combined'], r['weighted'])
            if best_v3 > r['std'] + 1e-6:
                improvements.append(r)

        if improvements:
            print(f"\n  Configs where v₃ beats standard FV ({len(improvements)}):")
            for r in improvements[:10]:
                best_v3 = max(r['switch'], r['combined'], r['weighted'])
                print(f"    (n={r['n']},m={r['m']}): std={r['std']:.4f} -> "
                      f"v3_best={best_v3:.4f} (+{best_v3-r['std']:.4f}) "
                      f"| FV_db={r['fv_db']:.4f}")


if __name__ == "__main__":
    main()
