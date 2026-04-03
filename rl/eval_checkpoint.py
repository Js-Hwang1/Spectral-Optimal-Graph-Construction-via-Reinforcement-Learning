#!/usr/bin/env python3
"""
Evaluate checkpoint against baselines database.

Compares GNN edge selection RL agent against FV, ER, SW baselines for ALL (n, m) datapoints.
Uses cached JSON first; falls back to HuggingFace DB for missing n values.

Inference: GNN(D⁻¹A) + degree/2-hop/3-hop features only (no eigensolvers). O(N²) per step.

Usage:
    python3 eval_checkpoint.py logs/train_.../best_policy.pt --n 8,10,12,14,16 --trials 10
    python3 eval_checkpoint.py checkpoint.pt --n 12,16 --trials 5
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.gnm_env import (
    GNMEnv, GNMConfig,
    compute_node_features, compute_graph_features, compute_R,
    algebraic_connectivity,
    K_STEPS,
)
from models.gnm_policy import EdgeSelectionPolicy, PolicyConfig

# --- Baseline loading: cache first, HF fallback ---

_cache_path = Path(__file__).parent / 'baselines_cache.json'
_cache_data = {}
if _cache_path.exists():
    with open(_cache_path) as f:
        _cache_data = json.load(f)

_hf_df = None  # lazy-loaded HF dataframe


def _load_hf_for_n(n: int) -> dict:
    """Fetch baselines for a given n from HuggingFace and cache them."""
    global _hf_df, _cache_data

    if _hf_df is None:
        try:
            from datasets import load_dataset
            print("  Downloading baselines from HuggingFace (first time)...")
            ds = load_dataset("June30916/algebraic-connectivity-baselines", split="train")
            _hf_df = ds.to_pandas()
            print(f"  Loaded {len(_hf_df)} rows from HF dataset")
        except Exception as e:
            print(f"  WARNING: Failed to load HF dataset: {e}")
            _hf_df = False
            return {}

    if _hf_df is False:
        return {}

    # Filter for this n
    df_n = _hf_df[_hf_df['n'] == n]
    if len(df_n) == 0:
        return {}

    # Build dict: {m: {fv: ..., er: ..., sw_025: ..., sw_050: ..., sw_075: ...}}
    n_data = {}
    cols = ['fv', 'er', 'sw_025', 'sw_050', 'sw_075']
    for _, row in df_n.iterrows():
        m_key = str(int(row['m']))
        entry = {}
        for c in cols:
            if c in row and not (isinstance(row[c], float) and np.isnan(row[c])):
                entry[c] = float(row[c])
        if entry:
            n_data[m_key] = entry

    # Update cache in memory and on disk
    _cache_data[str(n)] = n_data
    try:
        with open(_cache_path, 'w') as f:
            json.dump(_cache_data, f)
        print(f"  Cached n={n} ({len(n_data)} m-values) to {_cache_path.name}")
    except Exception as e:
        print(f"  WARNING: Failed to write cache: {e}")

    return n_data


def get_baselines_batch(n, m_list):
    """Get baselines for (n, m_list). Uses cache first, HF DB as fallback."""
    n_key = str(n)

    # Try cache
    if n_key in _cache_data:
        n_data = _cache_data[n_key]
        result = {int(m): n_data[str(m)] for m in m_list if str(m) in n_data}
        if result:
            return result

    # Fallback: fetch from HF and cache
    n_data = _load_hf_for_n(n)
    if n_data:
        return {int(m): n_data[str(m)] for m in m_list if str(m) in n_data}

    return {}


def _eval_single_config(args, policy_state_dict, policy_config_dict, num_trials, max_n):
    """Worker function: evaluate GNN edge selection agent on a single (n, m) config."""
    n, m, baselines = args

    policy_config = PolicyConfig(**policy_config_dict)
    policy = EdgeSelectionPolicy(policy_config)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    rl_lambda2 = []
    rl_rewires = []

    for trial_seed in range(num_trials):
        gnm_config = GNMConfig(min_n=n, max_n=n)
        env = GNMEnv(gnm_config, seed=trial_seed, eval_mode=True)
        env.inference_mode = True
        env.reset(n=n, m=m)

        R = compute_R(n, m)

        for k in range(K_STEPS):
            step_frac = k / K_STEPS

            # Compute features — O(N²), no eigensolvers
            node_feat = compute_node_features(env.adj, env.degrees, n)
            edges = env.get_edge_index()
            non_edges = env.get_non_edge_index()
            graph_feat = compute_graph_features(n, m, step_frac, max_n)

            # Convert to tensors
            nf_t = torch.tensor(node_feat, dtype=torch.float32)
            adj_t = torch.tensor(env.adj, dtype=torch.float32)
            edges_t = torch.tensor(edges, dtype=torch.long)
            ne_t = torch.tensor(non_edges, dtype=torch.long)
            gf_t = torch.tensor(graph_feat, dtype=torch.float32)

            # Policy forward → GNN encode + deterministic edge selection
            with torch.no_grad():
                remove_idx, add_idx, _, _ = policy.act(
                    nf_t, adj_t, edges_t, ne_t, gf_t, R, deterministic=True
                )

            actual_R = len(remove_idx)
            if actual_R > 0:
                rem_list = [(int(edges[i, 0]), int(edges[i, 1])) for i in remove_idx.cpu()]
                add_list = [(int(non_edges[i, 0]), int(non_edges[i, 1])) for i in add_idx.cpu()]
                env.spectral_step(rem_list, add_list)

        rl_lambda2.append(algebraic_connectivity(env.adj))
        rl_rewires.append(env.total_rewires)

    max_m = n * (n - 1) // 2
    rl_mean = np.mean(rl_lambda2)
    rl_std = np.std(rl_lambda2)

    result = {
        'n': n, 'm': m,
        'density': m / max_m,
        'rl_mean': rl_mean,
        'rl_std': rl_std,
        'avg_rewires': np.mean(rl_rewires),
    }

    for key in ['fv', 'er', 'sw_025', 'sw_050', 'sw_075']:
        result[key] = baselines.get(key, None)

    best_baseline = max(
        (baselines.get(k, 0) or 0)
        for k in ['fv', 'er', 'sw_025', 'sw_050', 'sw_075']
    )
    result['best_baseline'] = best_baseline
    result['win'] = rl_mean > best_baseline + 1e-6 if best_baseline > 0 else False

    return result


def eval_checkpoint(checkpoint_path: str, n_values: list,
                    num_trials: int = 5, num_workers: int = None):
    """Evaluate checkpoint against baselines for given n values."""

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
        episode = checkpoint.get('episode', '?')
        metrics = checkpoint.get('metrics', {})
        config = checkpoint.get('config', {})
        print(f"Checkpoint from episode {episode}")
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
    else:
        state_dict = checkpoint
        config = {}
        print("Raw state dict (no metadata)")

    # Build policy config from checkpoint or defaults
    policy_config_dict = {
        'hidden_dim': config.get('hidden_dim', 64),
        'num_gnn_layers': config.get('num_gnn_layers', 3),
    }
    print(f"  Policy config: {policy_config_dict}")

    max_n = max(n_values)

    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    # Build task list for each requested n
    n_str = ','.join(str(n) for n in n_values)
    print(f"\nLoading baselines for n={n_str}...")
    tasks = []
    total_configs = 0
    sources = {}

    for n in n_values:
        max_m = n * (n - 1) // 2
        min_m = n - 1
        all_m = list(range(min_m, max_m + 1))

        n_key = str(n)
        source = "cache" if n_key in _cache_data else "HF"
        batch = get_baselines_batch(n, all_m)

        if not batch:
            print(f"  n={n}: no baselines found (cache or HF), skipping")
            continue

        sources[n] = source
        for m in all_m:
            baselines = batch.get(m, {})
            if not baselines:
                continue
            tasks.append((n, m, baselines))
            total_configs += 1

    for n in sorted(sources):
        count = sum(1 for t in tasks if t[0] == n)
        print(f"  n={n}: {count} configs ({sources[n]})")

    print(f"  Total: {total_configs} configs, {num_trials} trials each, {num_workers} workers")
    print(f"  Comparing against: FV, ER, SW(0.25), SW(0.50), SW(0.75)")

    # Run parallel evaluation
    print(f"\nRunning evaluation...")
    worker_fn = partial(
        _eval_single_config,
        policy_state_dict=state_dict,
        policy_config_dict=policy_config_dict,
        num_trials=num_trials,
        max_n=max_n,
    )

    results = []
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, tasks)):
            results.append(result)
            if (i + 1) % 10 == 0 or i + 1 == total_configs:
                print(f"  {i+1}/{total_configs} configs evaluated...", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    def _v(val):
        return f"{val:>9.4f}" if val is not None and val > 0 else f"{'---':>9}"

    # Full per-(n,m) table
    wins = sum(1 for r in results if r['win'])
    total = len(results)

    print(f"\n{'='*110}")
    print(f"RESULTS: GNN Edge Selection RL Agent vs ALL Baselines ({num_trials} trials/config)")
    print(f"{'='*110}")

    hdr = f"{'(n,m)':>9}  {'FV':>9} {'ER':>9} {'SW.25':>9} {'SW.50':>9} {'SW.75':>9} {'RL':>9} {'WIN':>5}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for r in results:
        fv  = r.get('fv')  or 0
        er  = r.get('er')  or 0
        s25 = r.get('sw_025') or 0
        s50 = r.get('sw_050') or 0
        s75 = r.get('sw_075') or 0
        rl  = r['rl_mean']
        tag = "  WIN" if r['win'] else ""
        label = f"({r['n']},{r['m']})"
        print(f"{label:>9}  {_v(fv)} {_v(er)} {_v(s25)} {_v(s50)} {_v(s75)} {rl:>9.4f} {tag}")

    print("-" * 110)

    # Summary
    print(f"\nSUMMARY: {wins}/{total} WINS (RL beats ALL baselines)")

    # Threshold analysis: how many configs reach 95% / 99% of best baseline
    reach_95 = sum(1 for r in results if r['best_baseline'] > 0 and r['rl_mean'] >= 0.95 * r['best_baseline'])
    reach_99 = sum(1 for r in results if r['best_baseline'] > 0 and r['rl_mean'] >= 0.99 * r['best_baseline'])
    with_baseline = sum(1 for r in results if r['best_baseline'] > 0)
    print(f"  >= 99% of best baseline: {reach_99}/{with_baseline}")
    print(f"  >= 95% of best baseline: {reach_95}/{with_baseline}")

    # Per-n breakdown
    for n in sorted(set(r['n'] for r in results)):
        nr = [r for r in results if r['n'] == n]
        nw = sum(1 for r in nr if r['win'])
        n95 = sum(1 for r in nr if r['best_baseline'] > 0 and r['rl_mean'] >= 0.95 * r['best_baseline'])
        n99 = sum(1 for r in nr if r['best_baseline'] > 0 and r['rl_mean'] >= 0.99 * r['best_baseline'])
        avg_rl = np.mean([r['rl_mean'] for r in nr])
        avg_best = np.mean([r['best_baseline'] for r in nr if r['best_baseline'] > 0])
        print(f"  n={n}: {nw}/{len(nr)} wins, {n99} @99%, {n95} @95%, avg RL={avg_rl:.3f}, avg best={avg_best:.3f}")

    # Density breakdown
    density_bins = [(0, 0.2, "sparse"), (0.2, 0.4, "low-mid"),
                    (0.4, 0.6, "mid"), (0.6, 0.8, "high-mid"), (0.8, 1.01, "dense")]

    print(f"\nDENSITY BREAKDOWN:")
    print(f"{'Range':<10} {'configs':>7} {'wins':>6} {'@99%':>6} {'@95%':>6} {'avg RL/best':>12}")
    print("-" * 55)
    for lo, hi, label in density_bins:
        br = [r for r in results if lo <= r['density'] < hi]
        if not br:
            continue
        bw = sum(1 for r in br if r['win'])
        b95 = sum(1 for r in br if r['best_baseline'] > 0 and r['rl_mean'] >= 0.95 * r['best_baseline'])
        b99 = sum(1 for r in br if r['best_baseline'] > 0 and r['rl_mean'] >= 0.99 * r['best_baseline'])
        ratios = [r['rl_mean'] / r['best_baseline'] * 100 for r in br if r['best_baseline'] > 0]
        avg_r = np.mean(ratios) if ratios else 0
        print(f"{label:<10} {len(br):>7} {bw:>6} {b99:>6} {b95:>6} {avg_r:>11.1f}%")

    print(f"{'='*110}")

    # Save CSV
    csv_path = Path(checkpoint_path).parent / f"eval_n{n_str.replace(',', '_')}.csv"
    with open(csv_path, 'w') as f:
        f.write("n,m,density,rl,rl_std,fv,er,sw025,sw050,sw075,win\n")
        for r in results:
            f.write(f"{r['n']},{r['m']},{r['density']:.4f},"
                    f"{r['rl_mean']:.6f},{r['rl_std']:.6f},"
                    f"{r.get('fv','')},{r.get('er','')},"
                    f"{r.get('sw_025','')},{r.get('sw_050','')},{r.get('sw_075','')},"
                    f"{1 if r['win'] else 0}\n")
    print(f"\nCSV saved: {csv_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoint against baselines")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--n", type=str, required=True,
                        help="Comma-separated n values (e.g. 8,10,12,14,16)")
    parser.add_argument("--trials", type=int, default=5, help="Trials per (n,m) config")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: cpu_count)")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]
    eval_checkpoint(args.checkpoint, n_values=n_values,
                    num_trials=args.trials, num_workers=args.workers)
