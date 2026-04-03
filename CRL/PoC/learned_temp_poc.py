#!/usr/bin/env python3
"""
PoC: Learned Temperature — RL learns when to trust vs distrust FV's gradient.

Hypothesis: FV fails not because its gradient is wrong, but because it follows
it too greedily. A "temperature" parameter controls greediness:
  - Low temp (τ→0): greedy argmax (= standard FV)
  - High temp (τ→∞): uniform random (= exploration)

Instead of learning the full edge scoring, the RL agent learns ONLY a scalar
temperature τ(state) that modulates FV's exact gradient. This is a much simpler
function that should generalize better.

This PoC tests static temperatures (no learning) to establish:
  1. Does non-zero temperature ever beat greedy FV? (feasibility)
  2. What temperature range works? (hyperparameter guidance)
  3. Is the optimal temperature density-dependent? (what RL should learn)

Method:
  1. Compute FV scores: s(i,j) = (v₂[i] - v₂[j])²
  2. Apply softmax with temperature: p(i,j) ∝ exp(s(i,j) / τ)
  3. Sample edge from distribution (not argmax)
  4. Multiple samples per config, take best (simulates multiple RL rollouts)

Temperatures tested: 0 (greedy), 0.001, 0.01, 0.05, 0.1, 0.5
Initialization: Random spanning tree, 5 seeds × 3 samples = 15 rollouts, best-of.

Usage:
    python learned_temp_poc.py --n 16 --workers 8
    python learned_temp_poc.py --n 16,24 --workers 8
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


# ---------- Temperature FV ----------

def temp_fv(n, m, tau, seed=0):
    """
    FV construction with softmax temperature on Fiedler gap scores.

    Args:
        tau: Temperature. 0 = greedy argmax (standard FV).
             Higher = more stochastic exploration.
    """
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    for _ in range(edges_to_add):
        L = np.diag(adj.sum(axis=1)) - adj
        _, evecs = linalg.eigh(L)
        v2 = evecs[:, 1]

        diff = v2[:, None] - v2[None, :]
        scores = diff ** 2

        mask = (adj == 0) & upper_tri
        if not mask.any():
            break

        valid_idx = np.where(mask.ravel())[0]
        valid_scores = scores.ravel()[valid_idx]

        if tau <= 0 or tau < 1e-8:
            # Greedy: argmax
            best = valid_idx[np.argmax(valid_scores)]
        else:
            # Softmax sampling with temperature
            # Normalize scores to prevent overflow
            logits = valid_scores / tau
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            best = valid_idx[rng.choice(len(valid_idx), p=probs)]

        i, j = best // n, best % n
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


# ---------- Worker ----------

TAUS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
NUM_SEEDS = 5
SAMPLES_PER_SEED = 3  # for stochastic methods, run multiple samples per seed


def _eval_config(args):
    n, m = args
    fv_bl = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    results = {'n': n, 'm': m, 'fv_db': fv_bl, 'best_bl': best_bl}

    for tau in TAUS:
        if tau <= 0:
            # Greedy is deterministic — just vary the seed (spanning tree)
            best = max(
                temp_fv(n, m, tau=0, seed=s)
                for s in range(NUM_SEEDS)
            )
        else:
            # Stochastic — vary both seed and sample
            best = max(
                temp_fv(n, m, tau=tau, seed=s * 1000 + sample)
                for s in range(NUM_SEEDS)
                for sample in range(SAMPLES_PER_SEED)
            )
        results[f't{tau}'] = best

    return results


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Learned Temperature PoC: softmax temperature on FV gradient")
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

    print(f"Learned Temperature PoC | n={','.join(str(n) for n in n_values)} | "
          f"{len(tasks)} configs | {args.workers} workers")
    print(f"Temperatures: {TAUS}")
    print(f"Seeds: {NUM_SEEDS}, Samples/seed: {SAMPLES_PER_SEED} "
          f"(greedy: {NUM_SEEDS} rollouts, stochastic: "
          f"{NUM_SEEDS * SAMPLES_PER_SEED} rollouts)")
    print(f"(τ=0 is standard greedy FV baseline)\n")

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    tau_keys = [f't{t}' for t in TAUS]

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
        print(f"  {'-'*62}")

        for tk, tau in zip(tau_keys, TAUS):
            avg = np.mean([r[tk] for r in nr])
            ratio_fv = avg / avg_fv * 100 if avg_fv > 0 else 0
            ratio_best = avg / avg_best * 100 if avg_best > 0 else 0
            wins_fv = sum(1 for r in nr if r[tk] > r['fv_db'] + 1e-6
                          and r['fv_db'] > 0)
            ties_fv = sum(1 for r in nr if abs(r[tk] - r['fv_db']) < 1e-6
                          and r['fv_db'] > 0)
            wins_best = sum(1 for r in nr if r[tk] > r['best_bl'] + 1e-6
                            and r['best_bl'] > 0)
            label = f"τ={tau}" + (" (greedy)" if tau == 0 else "")
            print(f"  {label:<18} {avg:8.3f} {ratio_fv:8.1f}% "
                  f"{wins_fv:4d} {ties_fv:3d} {ratio_best:8.1f}% {wins_best:4d}")

        # Per-config: best tau>0 vs greedy
        print(f"\n  Per-config best τ>0 vs greedy (τ=0):")
        nongreedy_keys = [k for k in tau_keys if k != 't0.0']
        improved = 0
        total_gain = 0.0
        for r in nr:
            best_ng = max(r[k] for k in nongreedy_keys)
            if best_ng > r['t0.0'] + 1e-6:
                improved += 1
                total_gain += best_ng - r['t0.0']
        hurt = sum(1 for r in nr
                    if max(r[k] for k in nongreedy_keys) < r['t0.0'] - 1e-6)
        print(f"    Improved: {improved}/{len(nr)} configs "
              f"(+{total_gain:.3f} total λ₂ gain)")
        print(f"    Same or worse: {len(nr) - improved}/{len(nr)}")

        # Breakdown by density
        print(f"\n  Best temperature by density:")
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
            best_tau = None
            best_avg = 0
            for tk, tau in zip(tau_keys, TAUS):
                avg = np.mean([r[tk] for r in subset])
                if avg > best_avg:
                    best_avg = avg
                    best_tau = tau
            std_avg = np.mean([r['t0.0'] for r in subset])
            bl_avg = np.mean([r['best_bl'] for r in subset
                              if r['best_bl'] > 0])
            delta = best_avg - std_avg
            print(f"    {label}: best τ={best_tau:.3f} "
                  f"(avg={best_avg:.3f}, greedy={std_avg:.3f}, "
                  f"Δ={delta:+.3f}, vs_bl={best_avg/bl_avg*100:.1f}%)")

        # Oracle: per-config best temperature
        print(f"\n  Oracle (per-config best τ):")
        oracle_vals = [max(r[tk] for tk in tau_keys) for r in nr]
        oracle_avg = np.mean(oracle_vals)
        oracle_wins_bl = sum(1 for r, ov in zip(nr, oracle_vals)
                             if ov > r['best_bl'] + 1e-6 and r['best_bl'] > 0)
        print(f"    avg={oracle_avg:.3f} | vs FV_db: "
              f"{oracle_avg/avg_fv*100:.1f}% | "
              f"vs best: {oracle_avg/avg_best*100:.1f}% ({oracle_wins_bl}W)")
        print(f"    (Upper bound if RL perfectly learns τ per config)")


if __name__ == "__main__":
    main()
