#!/usr/bin/env python3
"""
PoC: K=100 steps, RR_K=8, M//10 swaps per step at n=1024.

Simulates a full episode:
  For each of K=100 steps:
    1. Lanczos refresh (simulated by exact eigendecomp) → fresh RR subspace
    2. M//10 random swaps with RR updates only
    3. Check: how much did RR drift by end of step?

Key question: Does fresh Lanczos at each step start recover from drift?
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
    """Random swap. Returns (added, removed) or (None, None)."""
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

    adj[ai, aj] = 0.0
    adj[aj, ai] = 0.0
    degrees[ai] -= 1
    degrees[aj] -= 1
    return None, None


def simulate_episode(n, m, K, k_rr, swap_frac, seed=42):
    """Simulate K steps with Lanczos refresh + M*swap_frac swaps per step."""
    rng = np.random.RandomState(seed)
    adj = build_ring_random(n, m, rng)
    degrees = adj.sum(axis=1).copy()
    rng_swap = np.random.RandomState(seed + 1000)

    num_swaps_per_step = max(1, int(m * swap_frac))

    step_data = []

    for step in range(K):
        # === LANCZOS REFRESH (exact eigendecomp) ===
        vals, vecs = exact_spectrum(adj)
        n_eig = min(k_rr, n - 1)
        V = vecs[:, 1:1 + n_eig].copy()
        lams = vals[1:1 + n_eig].copy()

        exact_lam2_start = vals[1]
        rr_lam2_start = float(lams[0])
        cos_start = cosine_sim(V[:, 0], vecs[:, 1])

        # === M//10 SWAPS WITH RR ONLY ===
        swaps_done = 0
        # Check at 25%, 50%, 75%, 100%
        checkpoints = {
            int(num_swaps_per_step * f): f
            for f in [0.25, 0.5, 0.75, 1.0]
        }
        mid_checks = {}

        for s in range(num_swaps_per_step):
            added, removed = do_one_swap(adj, degrees, n, rng_swap)
            if added is None:
                continue

            ai, aj = added
            ri, rj = removed
            V, lams = rr_update(V, lams, ai, aj, sign=+1.0)
            V, lams = rr_update(V, lams, ri, rj, sign=-1.0)
            swaps_done += 1

            if swaps_done in checkpoints:
                exact_vals_mid, exact_vecs_mid = exact_spectrum(adj)
                mid_checks[checkpoints[swaps_done]] = {
                    'rel_err': abs(float(lams[0]) - exact_vals_mid[1]) / max(abs(exact_vals_mid[1]), 1e-10),
                    'cos': cosine_sim(V[:, 0], exact_vecs_mid[:, 1]),
                }

        # === END OF STEP: exact check ===
        vals_end, vecs_end = exact_spectrum(adj)
        exact_lam2_end = vals_end[1]
        rr_lam2_end = float(lams[0])
        cos_end = cosine_sim(V[:, 0], vecs_end[:, 1])
        rel_err_end = abs(rr_lam2_end - exact_lam2_end) / max(abs(exact_lam2_end), 1e-10)

        step_data.append({
            'step': step,
            'swaps': swaps_done,
            'exact_lam2': exact_lam2_end,
            'rr_lam2_end': rr_lam2_end,
            'rel_err_end': rel_err_end,
            'cos_end': cos_end,
            'cos_start': cos_start,
            'mid_checks': mid_checks,
        })

    return step_data


def run_config(n, m, K, k_rr, swap_frac, label):
    """Run and print results for one configuration."""
    print(f"\n{'─' * 85}")
    print(f"  {label}")
    print(f"  n={n}, m={m}, K={K}, RR_K={k_rr}, "
          f"swaps/step={max(1, int(m * swap_frac))}")
    print(f"{'─' * 85}")

    t0 = time.time()
    data = simulate_episode(n, m, K, k_rr, swap_frac, seed=42)
    elapsed = time.time() - t0

    # Print summary: every 10th step
    print(f"\n  {'Step':>5} | {'Swaps':>5} | {'exact λ₂':>9} | "
          f"{'@25% err':>8} {'cos':>5} | "
          f"{'@50% err':>8} {'cos':>5} | "
          f"{'@100% err':>9} {'cos':>5}")
    print(f"  {'─' * 78}")

    for d in data:
        step = d['step']
        if step % 10 != 0 and step != K - 1 and step != len(data) - 1:
            continue

        mc = d['mid_checks']
        c25 = mc.get(0.25, {})
        c50 = mc.get(0.5, {})

        s25 = f"{c25.get('rel_err', 0):>7.1%} {c25.get('cos', 0):>5.3f}" if c25 else f"{'---':>7} {'---':>5}"
        s50 = f"{c50.get('rel_err', 0):>7.1%} {c50.get('cos', 0):>5.3f}" if c50 else f"{'---':>7} {'---':>5}"

        print(f"  {step:>5} | {d['swaps']:>5} | {d['exact_lam2']:>9.4f} | "
              f"{s25} | {s50} | "
              f"{d['rel_err_end']:>8.1%} {d['cos_end']:>5.3f}")

    # Aggregate stats
    errs_end = [d['rel_err_end'] for d in data]
    cos_end = [d['cos_end'] for d in data]
    errs_50 = [d['mid_checks'].get(0.5, {}).get('rel_err', None) for d in data]
    cos_50 = [d['mid_checks'].get(0.5, {}).get('cos', None) for d in data]
    errs_50 = [e for e in errs_50 if e is not None]
    cos_50 = [c for c in cos_50 if c is not None]

    print(f"\n  AGGREGATE ({len(data)} steps):")
    print(f"    End-of-step:  mean err={np.mean(errs_end):.1%}, "
          f"mean cos={np.mean(cos_end):.3f}, "
          f"min cos={np.min(cos_end):.3f}")
    if errs_50:
        print(f"    Mid-step @50%: mean err={np.mean(errs_50):.1%}, "
              f"mean cos={np.mean(cos_50):.3f}, "
              f"min cos={np.min(cos_50):.3f}")
    print(f"    λ₂ trajectory: {data[0]['exact_lam2']:.4f} → {data[-1]['exact_lam2']:.4f}")
    print(f"    Time: {elapsed:.1f}s")


def main():
    print("=" * 85)
    print("PoC: K=100, RR_K=8, M//10 swaps/step — Full Episode at N=1024")
    print("=" * 85)

    # Quick sanity at small n
    run_config(64, 192, 100, 8, 0.1, "n=64 sparse (m=3n)")
    run_config(64, 512, 100, 8, 0.1, "n=64 medium (m=n√n)")

    # Target: n=1024
    run_config(256, 768, 100, 8, 0.1, "n=256 sparse (m=3n)")
    run_config(256, 4096, 100, 8, 0.1, "n=256 medium (m=n√n)")

    run_config(1024, 3072, 100, 8, 0.1, "n=1024 sparse (m=3n)")
    run_config(1024, 32768, 100, 8, 0.1, "n=1024 medium (m=n√n)")


if __name__ == "__main__":
    main()
