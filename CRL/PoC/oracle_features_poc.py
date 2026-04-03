#!/usr/bin/env python3
"""
PoC: Oracle Feature Analysis — What features predict the true best edge?

=== Question ===
FV uses v₂ gap² as its scoring. Can OTHER features predict the oracle's
choice better? If yes → directional signal exists and is learnable.
If no → the deviation from FV is fundamentally random (probabilistic).

=== Key insight: Second-order perturbation theory ===
FV uses the FIRST-ORDER perturbation of λ₂:
    Δλ₂⁽¹⁾ = (v₂[i] - v₂[j])²

But the SECOND-ORDER correction is:
    Δλ₂⁽²⁾ = (v₂[i]-v₂[j])² × Σ_{k≥3} (v_k[i]-v_k[j])² / (λ₂ - λ_k)

This is always NEGATIVE (λ_k > λ₂ for k≥3), meaning FV OVERESTIMATES
the benefit of edges that have large higher-eigenvector gaps. The combined
first+second order prediction should correlate better with true Δλ₂.

=== Features tested ===
  1. v₂ gap²                    (= FV, first-order)
  2. 1st + 2nd order perturb    (perturbation theory)
  3. v₃ gap²
  4. max(v₂, v₃) gap²
  5. Σ v_k gap² (k=2..K)       (cascade)
  6. degree sum
  7. degree product

For each: Spearman rank correlation with true Δλ₂, and full rollout.

Usage:
    python oracle_features_poc.py --n 16 --workers 8
    python oracle_features_poc.py --n 16,24 --workers 8
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import linalg, stats
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


# ===== Feature scorings =====

K_EIGVECS = 8  # v₂ through v₉


def compute_all_features(adj, n, evals, evecs, valid_idx):
    """
    Compute multiple edge features for all candidate non-edges.
    Returns dict of feature_name → array of scores.
    """
    v2 = evecs[:, 1]
    lam2 = evals[1]
    deg = adj.sum(axis=1)
    k_use = min(K_EIGVECS, n - 1)

    features = {}

    # 1. v₂ gap² (= FV first-order)
    v2_gaps = np.array([(v2[f // n] - v2[f % n]) ** 2 for f in valid_idx])
    features['v2_gap'] = v2_gaps

    # 2. Second-order perturbation correction
    # Δλ₂⁽²⁾ = (v₂ gap)² × Σ_{k≥3} (v_k gap)² / (λ₂ - λ_k)
    second_order = np.zeros(len(valid_idx))
    for k in range(2, k_use):
        vk = evecs[:, k + 1]
        lam_k = evals[k + 1]
        denom = lam2 - lam_k  # negative for k > 2
        if abs(denom) < 1e-12:
            continue
        vk_gaps = np.array([(vk[f // n] - vk[f % n]) ** 2 for f in valid_idx])
        second_order += vk_gaps / denom
    # Combined: first + second order
    features['perturb_2nd'] = v2_gaps * (1.0 + v2_gaps * second_order)

    # 3. v₃ gap²
    if n > 2:
        v3 = evecs[:, 2]
        features['v3_gap'] = np.array([
            (v3[f // n] - v3[f % n]) ** 2 for f in valid_idx])

    # 4. max(v₂, v₃) gap²
    if n > 2:
        features['max_v2v3'] = np.maximum(
            features['v2_gap'], features['v3_gap'])

    # 5. sum v_k gap² for degenerate block
    cascade = np.zeros(len(valid_idx))
    for k in range(k_use):
        vk = evecs[:, k + 1]
        lam_k = evals[k + 1]
        if k > 0:
            gap = (lam_k - evals[k]) / max(abs(evals[k]), 1e-8)
            if gap > 0.3:
                break  # stop at first large gap
        vk_gaps = np.array([(vk[f // n] - vk[f % n]) ** 2 for f in valid_idx])
        cascade += vk_gaps
    features['cascade'] = cascade

    # 6. degree sum (normalized)
    features['deg_sum'] = np.array([
        (deg[f // n] + deg[f % n]) / (2 * (n - 1))
        for f in valid_idx])

    # 7. degree product (normalized)
    features['deg_prod'] = np.array([
        deg[f // n] * deg[f % n] / (n - 1) ** 2
        for f in valid_idx])

    # 8. v₂ gap × (1 - deg_sum) — prefer connecting low-degree nodes
    features['v2_lowdeg'] = features['v2_gap'] * (
        1.0 - features['deg_sum'] + 0.01)

    return features


# ===== Per-step correlation analysis =====

def analyze_correlations(n, m, seed):
    """Run FV, compute features + oracle at each step, return correlations."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    if edges_to_add <= 0:
        return []

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

        if len(valid_idx) < 3:
            # Not enough candidates for meaningful correlation
            # Just follow FV
            fv_scores = np.array([
                (v2[f // n] - v2[f % n]) ** 2 for f in valid_idx])
            chosen = valid_idx[np.argmax(fv_scores)]
            i, j = chosen // n, chosen % n
            adj[i, j] = adj[j, i] = 1
            continue

        # All features
        features = compute_all_features(adj, n, evals, evecs, valid_idx)

        # Oracle Δλ₂
        oracle_deltas = np.zeros(len(valid_idx))
        for k, flat in enumerate(valid_idx):
            i, j = flat // n, flat % n
            adj[i, j] = adj[j, i] = 1
            oracle_deltas[k] = float(linalg.eigvalsh(
                np.diag(adj.sum(axis=1)) - adj)[1]) - lam2
            adj[i, j] = adj[j, i] = 0

        # Spearman correlation of each feature with oracle
        corrs = {}
        for name, scores in features.items():
            if np.std(scores) < 1e-12 or np.std(oracle_deltas) < 1e-12:
                corrs[name] = 0.0
            else:
                rho, _ = stats.spearmanr(scores, oracle_deltas)
                corrs[name] = float(rho) if not np.isnan(rho) else 0.0

        # Which feature's argmax matches oracle?
        oracle_choice = np.argmax(oracle_deltas)
        hits = {}
        for name, scores in features.items():
            hits[name] = int(np.argmax(scores) == oracle_choice)

        gap_ratio = (evals[2] - lam2) / max(lam2, 1e-8) if lam2 > 1e-8 else 999
        density = (n - 1 + step) / (n * (n - 1) // 2)

        steps.append({
            'corrs': corrs,
            'hits': hits,
            'gap_ratio': float(gap_ratio),
            'density': density,
        })

        # Follow FV for construction
        fv_choice = np.argmax(features['v2_gap'])
        chosen = valid_idx[fv_choice]
        i, j = chosen // n, chosen % n
        adj[i, j] = adj[j, i] = 1

    return steps


# ===== Full rollout with different scorings =====

NUM_SEEDS = 3


def build_with_scoring(n, m, scoring_name, seed):
    """Build graph using a specific feature scoring."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    for step in range(edges_to_add):
        L = np.diag(adj.sum(axis=1)) - adj
        evals, evecs = linalg.eigh(L)

        mask = (adj == 0) & upper_tri
        if not mask.any():
            break
        valid_idx = np.where(mask.ravel())[0]

        if scoring_name == 'oracle':
            lam2 = evals[1]
            deltas = np.zeros(len(valid_idx))
            for k, flat in enumerate(valid_idx):
                i, j = flat // n, flat % n
                adj[i, j] = adj[j, i] = 1
                deltas[k] = float(linalg.eigvalsh(
                    np.diag(adj.sum(axis=1)) - adj)[1]) - lam2
                adj[i, j] = adj[j, i] = 0
            best_k = np.argmax(deltas)
        else:
            features = compute_all_features(adj, n, evals, evecs, valid_idx)
            scores = features.get(scoring_name, features['v2_gap'])
            best_k = np.argmax(scores)

        chosen = valid_idx[best_k]
        i, j = chosen // n, chosen % n
        adj[i, j] = adj[j, i] = 1

    return algebraic_connectivity(adj)


# ===== Worker =====

SCORINGS = ['v2_gap', 'perturb_2nd', 'v3_gap', 'max_v2v3',
            'cascade', 'v2_lowdeg', 'oracle']


def _eval_config(args):
    n, m = args
    fv_db = get_fv_baseline(n, m)
    best_bl = get_best_baseline(n, m)

    # Correlation analysis (1 seed)
    step_data = analyze_correlations(n, m, seed=0)

    # Rollouts (best of NUM_SEEDS)
    rollouts = {}
    for scoring in SCORINGS:
        rollouts[scoring] = max(
            build_with_scoring(n, m, scoring, s)
            for s in range(NUM_SEEDS))

    return {
        'n': n, 'm': m,
        'fv_db': fv_db, 'best_bl': best_bl,
        'steps': step_data,
        'rollouts': rollouts,
    }


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description="Oracle Feature Analysis: what features predict the oracle?")
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

    print(f"Oracle Feature PoC | n={args.n} | {len(tasks)} configs | "
          f"{args.workers} workers | {NUM_SEEDS} seeds")
    print(f"Features: {', '.join(SCORINGS[:-1])}")
    print()

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_config, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    feature_names = ['v2_gap', 'perturb_2nd', 'v3_gap', 'max_v2v3',
                     'cascade', 'v2_lowdeg']

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        max_m = n * (n - 1) // 2

        all_steps = []
        for r in nr:
            all_steps.extend(r['steps'])
        total_steps = len(all_steps)

        print(f"\n{'=' * 90}")
        print(f"N = {n} ({len(nr)} configs, {total_steps} steps)")
        print(f"{'=' * 90}")

        # --- 1. Average Spearman correlation with oracle ---
        print(f"\n  1. Spearman rank correlation with oracle Δλ₂:")
        print(f"     {'Feature':<20} {'ρ (all)':>8} {'ρ (degen)':>10} "
              f"{'ρ (clear)':>10} {'accuracy':>9}")
        print(f"     {'-' * 62}")

        for fname in feature_names:
            # All steps
            rhos = [s['corrs'].get(fname, 0) for s in all_steps
                    if fname in s['corrs']]
            avg_rho = np.mean(rhos) if rhos else 0

            # Degenerate (gap < 0.1)
            degen = [s['corrs'].get(fname, 0) for s in all_steps
                     if s['gap_ratio'] < 0.1 and fname in s['corrs']]
            rho_deg = np.mean(degen) if degen else 0

            # Clear (gap > 0.3)
            clear = [s['corrs'].get(fname, 0) for s in all_steps
                     if s['gap_ratio'] > 0.3 and fname in s['corrs']]
            rho_clr = np.mean(clear) if clear else 0

            # Accuracy (argmax matches oracle)
            hits = [s['hits'].get(fname, 0) for s in all_steps
                    if fname in s['hits']]
            acc = np.mean(hits) if hits else 0

            marker = " <--" if avg_rho > np.mean(
                [s['corrs'].get('v2_gap', 0) for s in all_steps]) + 0.005 else ""
            print(f"     {fname:<20} {avg_rho:8.4f} {rho_deg:10.4f} "
                  f"{rho_clr:10.4f} {acc * 100:8.1f}%{marker}")

        # --- 2. Correlation improvement over FV ---
        print(f"\n  2. Per-step: how often does each feature beat v₂ gap "
              f"correlation?")
        for fname in feature_names:
            if fname == 'v2_gap':
                continue
            better = sum(1 for s in all_steps
                         if fname in s['corrs'] and 'v2_gap' in s['corrs']
                         and s['corrs'][fname] > s['corrs']['v2_gap'] + 0.01)
            worse = sum(1 for s in all_steps
                        if fname in s['corrs'] and 'v2_gap' in s['corrs']
                        and s['corrs'][fname] < s['corrs']['v2_gap'] - 0.01)
            total_valid = sum(1 for s in all_steps
                              if fname in s['corrs'] and 'v2_gap' in s['corrs'])
            print(f"     {fname:<20}: better {better}/{total_valid} "
                  f"({better / max(total_valid, 1) * 100:.1f}%), "
                  f"worse {worse}/{total_valid} "
                  f"({worse / max(total_valid, 1) * 100:.1f}%)")

        # --- 3. Full rollout comparison ---
        print(f"\n  3. Full rollout (best-of-{NUM_SEEDS} seeds):")
        avg_fv_db = np.mean([r['fv_db'] for r in nr])
        avg_best = np.mean([r['best_bl'] for r in nr])

        scoring_labels = [
            ('v2_gap', 'v₂ gap (=FV)'),
            ('perturb_2nd', '1st+2nd order'),
            ('v3_gap', 'v₃ gap'),
            ('max_v2v3', 'max(v₂,v₃)'),
            ('cascade', 'cascade (degen)'),
            ('v2_lowdeg', 'v₂ × low-deg'),
            ('oracle', 'Oracle'),
        ]

        print(f"     {'Scoring':<20} {'avg λ₂':>8} {'vs FV_db':>9} "
              f"{'vs best':>9} {'vs FV_ST':>9}")
        print(f"     {'-' * 60}")

        fv_avg = np.mean([r['rollouts']['v2_gap'] for r in nr])
        for key, label in scoring_labels:
            avg = np.mean([r['rollouts'][key] for r in nr])
            r_fv_db = avg / avg_fv_db * 100 if avg_fv_db > 0 else 0
            r_best = avg / avg_best * 100 if avg_best > 0 else 0
            r_fv_st = avg / fv_avg * 100 if fv_avg > 0 else 0
            wins = sum(1 for r in nr
                       if r['rollouts'][key] > r['rollouts']['v2_gap'] + 1e-6)
            marker = " **" if avg > fv_avg + 0.01 else ""
            print(f"     {label:<20} {avg:8.3f} {r_fv_db:8.1f}% "
                  f"{r_best:8.1f}% {r_fv_st:8.1f}% "
                  f"({wins}W){marker}")

        # --- 4. Density breakdown for best non-FV scoring ---
        # Find best scoring
        best_scoring = None
        best_avg = fv_avg
        for key, _ in scoring_labels:
            if key == 'oracle' or key == 'v2_gap':
                continue
            avg = np.mean([r['rollouts'][key] for r in nr])
            if avg > best_avg:
                best_avg = avg
                best_scoring = key

        if best_scoring:
            print(f"\n  4. Best non-FV scoring ({best_scoring}) by density:")
            bins = [
                ("sparse  <0.3", 0.0, 0.3),
                ("medium 0.3-0.6", 0.3, 0.6),
                ("dense   ≥0.6", 0.6, 1.01),
            ]
            for label, lo, hi in bins:
                sub = [r for r in nr if lo <= r['m'] / max_m < hi]
                if not sub:
                    continue
                avg_f = np.mean([r['rollouts']['v2_gap'] for r in sub])
                avg_b = np.mean([r['rollouts'][best_scoring] for r in sub])
                avg_o = np.mean([r['rollouts']['oracle'] for r in sub])
                wins = sum(1 for r in sub
                           if r['rollouts'][best_scoring] >
                           r['rollouts']['v2_gap'] + 1e-6)
                print(f"     {label}: FV={avg_f:.3f} "
                      f"{best_scoring}={avg_b:.3f} "
                      f"oracle={avg_o:.3f} ({wins}W/{len(sub)})")
        else:
            print(f"\n  4. No scoring beats FV on average.")

        # --- 5. Key finding ---
        oracle_avg = np.mean([r['rollouts']['oracle'] for r in nr])
        print(f"\n  5. Summary:")
        print(f"     FV (spanning tree): {fv_avg:.3f}")
        print(f"     Oracle:             {oracle_avg:.3f} "
              f"(+{oracle_avg - fv_avg:.3f})")
        if best_scoring:
            print(f"     Best feature:       {best_avg:.3f} "
                  f"(closes {(best_avg - fv_avg) / max(oracle_avg - fv_avg, 1e-8) * 100:.0f}% "
                  f"of FV→oracle gap)")
        all_rhos = {fname: np.mean([s['corrs'].get(fname, 0)
                                    for s in all_steps])
                    for fname in feature_names}
        best_corr = max(all_rhos, key=all_rhos.get)
        print(f"     Best correlator:    {best_corr} "
              f"(ρ={all_rhos[best_corr]:.4f} vs FV's "
              f"ρ={all_rhos['v2_gap']:.4f})")


if __name__ == "__main__":
    main()
