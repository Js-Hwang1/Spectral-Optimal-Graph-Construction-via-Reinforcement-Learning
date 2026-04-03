#!/usr/bin/env python3
"""
PoC: Is RR_K=8 good enough at N=1024?

Tests RR subspace tracking accuracy after random swaps.
Shows degradation curve: accuracy at 10%, 20%, ..., 100% of M/10 swaps.
Tests sparse (m=3n) and medium (m=n√n) densities.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.spectral import rr_update, is_connected


def build_ring_random(n, m, rng):
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    remaining = m - n
    if remaining > 0:
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        rng.shuffle(non_edges)
        for i, j in non_edges[:remaining]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def exact_spectrum(adj):
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    vals, vecs = np.linalg.eigh(L)
    return vals, vecs


def cosine_sim(a, b):
    dot = np.abs(np.dot(a, b))
    return dot / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15)


def do_one_swap(adj, degrees, n, rng, max_tries=500):
    """Random add+remove swap. Returns (added, removed) or (None, None)."""
    # Find random non-edge
    ai, aj = -1, -1
    for _ in range(max_tries):
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            ai, aj = min(i, j), max(i, j)
            break
    if ai < 0:
        return None, None

    adj[ai, aj] = 1.0
    adj[aj, ai] = 1.0
    degrees[ai] += 1
    degrees[aj] += 1

    # Find random removable non-bridge edge
    ri, rj = -1, -1
    for _ in range(max_tries):
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j and adj[i, j] > 0:
            i2, j2 = min(i, j), max(i, j)
            if (i2, j2) == (ai, aj):
                continue
            adj[i2, j2] = 0.0
            adj[j2, i2] = 0.0
            if is_connected(adj):
                degrees[i2] -= 1
                degrees[j2] -= 1
                return (ai, aj), (i2, j2)
            adj[i2, j2] = 1.0
            adj[j2, i2] = 1.0

    # Undo add
    adj[ai, aj] = 0.0
    adj[aj, ai] = 0.0
    degrees[ai] -= 1
    degrees[aj] -= 1
    return None, None


def test_one_config(n, m, k_values, num_swaps, num_checks=10, seed=42):
    """Test RR accuracy with degradation curve."""
    check_points = [max(1, int(num_swaps * (i + 1) / num_checks))
                    for i in range(num_checks)]
    check_set = set(check_points)

    results = {}
    for k in k_values:
        if k >= n:
            continue

        # Fresh graph per k
        rng = np.random.RandomState(seed)
        adj = build_ring_random(n, m, rng)
        deg = adj.sum(axis=1).copy()
        rng_swap = np.random.RandomState(seed + 1000)

        # Init RR from exact
        vals, vecs = exact_spectrum(adj)
        n_eig = min(k, n - 1)
        V = vecs[:, 1:1 + n_eig].copy()
        lams = vals[1:1 + n_eig].copy()

        checks = []
        swaps_done = 0
        t_rr = 0.0

        while swaps_done < num_swaps:
            added, removed = do_one_swap(adj, deg, n, rng_swap)
            if added is None:
                # Can't swap, skip
                break

            ai, aj = added
            ri, rj = removed

            t0 = time.perf_counter()
            V, lams = rr_update(V, lams, ai, aj, sign=+1.0)
            V, lams = rr_update(V, lams, ri, rj, sign=-1.0)
            t_rr += time.perf_counter() - t0

            swaps_done += 1

            if swaps_done in check_set:
                exact_vals, exact_vecs = exact_spectrum(adj)
                exact_lam2 = exact_vals[1]
                exact_v2 = exact_vecs[:, 1]
                rr_lam2 = float(lams[0])
                rr_v2 = V[:, 0]

                rel_err = abs(rr_lam2 - exact_lam2) / max(abs(exact_lam2), 1e-10)
                cos = cosine_sim(rr_v2, exact_v2)
                ratio = rr_lam2 / max(exact_lam2, 1e-10)

                checks.append({
                    'swap': swaps_done,
                    'pct': 100 * swaps_done / num_swaps,
                    'exact_lam2': exact_lam2,
                    'rr_lam2': rr_lam2,
                    'rel_err': rel_err,
                    'ratio': ratio,
                    'cos_sim': cos,
                })

        results[k] = {
            'checks': checks,
            'swaps_done': swaps_done,
            'rr_time': t_rr,
        }

    return results


def main():
    print("=" * 90)
    print("PoC: RR_K Scaling — Is k=8 Good Enough at N=1024?")
    print("=" * 90)

    # Test configs: (n, m_description, m_func)
    densities = [
        ("sparse m=3n", lambda n: 3 * n),
        ("medium m=n√n", lambda n: int(n * np.sqrt(n))),
    ]

    n_values = [64, 128, 256, 512, 1024]
    k_values = [4, 8, 16, 32, 64]

    for density_name, m_func in densities:
        print(f"\n{'━' * 90}")
        print(f"  Density: {density_name}")
        print(f"{'━' * 90}")

        for n in n_values:
            m = m_func(n)
            max_m = n * (n - 1) // 2
            m = min(m, max_m)
            num_swaps = max(1, m // 10)

            # Filter k values that make sense
            valid_k = [k for k in k_values if k < n - 1]

            print(f"\n  n={n:>5}, m={m:>6}, swaps={num_swaps:>5}")

            t0 = time.time()
            results = test_one_config(n, m, valid_k, num_swaps,
                                       num_checks=10, seed=42)
            elapsed = time.time() - t0

            # Degradation curve: show accuracy at 10%, 50%, 100%
            print(f"  {'k':>4} | "
                  f"{'@10% err':>9} {'cos':>6} | "
                  f"{'@50% err':>9} {'cos':>6} | "
                  f"{'@100% err':>9} {'cos':>6} | "
                  f"{'RR μs':>7}")
            print(f"  {'─' * 75}")

            for k in sorted(results.keys()):
                r = results[k]
                checks = r['checks']
                if len(checks) < 3:
                    # Not enough check points
                    if checks:
                        c = checks[-1]
                        us = r['rr_time'] / max(r['swaps_done'], 1) * 1e6
                        print(f"  {k:>4} | "
                              f"{'---':>9} {'---':>6} | "
                              f"{'---':>9} {'---':>6} | "
                              f"{c['rel_err']:>8.1%} {c['cos_sim']:>6.3f} | "
                              f"{us:>6.0f}")
                    continue

                # Find checks closest to 10%, 50%, 100%
                c10 = checks[0]  # ~10%
                c50 = checks[len(checks) // 2]  # ~50%
                c100 = checks[-1]  # ~100%
                us = r['rr_time'] / max(r['swaps_done'], 1) * 1e6

                print(f"  {k:>4} | "
                      f"{c10['rel_err']:>8.1%} {c10['cos_sim']:>6.3f} | "
                      f"{c50['rel_err']:>8.1%} {c50['cos_sim']:>6.3f} | "
                      f"{c100['rel_err']:>8.1%} {c100['cos_sim']:>6.3f} | "
                      f"{us:>6.0f}")

            print(f"  ({elapsed:.1f}s)")

    print(f"\n{'=' * 90}")
    print("VERDICT")
    print(f"{'=' * 90}")
    print("k=8 sufficient if: <10% λ₂ error, >0.90 cos through M/10 swaps")
    print("If not: recommend adaptive k or more frequent Lanczos refreshes")


if __name__ == "__main__":
    main()
