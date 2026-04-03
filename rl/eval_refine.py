#!/usr/bin/env python3
"""
Evaluate REFINE checkpoint against baselines.

Batch-scored evaluation:
  Per K-step: score ONCE for ADD, pick top-K adds (argmax).
  Score ONCE for REMOVE, pick top-K removes (argmax).
  num_swaps = max(1, int(m * swap_frac)).

With swap_frac=0.1: 10% of edges per step → O(N³) for dense graphs.
Tracks WINS (RL > best baseline) and TIES (RL within tolerance of best).

Usage:
    python eval_refine.py checkpoints/best.pt --n 8 --trials 5
    python eval_refine.py checkpoints/best.pt --n 8,10,12,16 --trials 10 --swap-frac 0.1
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

from envs.refine_env import RefineEnv, RefineEnvConfig
from envs.gnm_env import algebraic_connectivity
from models.refine_policy import RefinePolicy, RefineConfig

# --- Baseline loading ---

_cache_path = Path(__file__).parent / 'baselines_cache.json'
_cache_data = {}
if _cache_path.exists():
    with open(_cache_path) as f:
        _cache_data = json.load(f)

_hf_df = None


def _load_hf_for_n(n):
    global _hf_df, _cache_data
    if _hf_df is None:
        try:
            from datasets import load_dataset
            print("  Downloading baselines from HuggingFace...")
            ds = load_dataset(
                "June30916/algebraic-connectivity-baselines", split="train")
            _hf_df = ds.to_pandas()
            print(f"  Loaded {len(_hf_df)} rows")
        except Exception as e:
            print(f"  WARNING: Failed to load HF dataset: {e}")
            _hf_df = False
            return {}
    if _hf_df is False:
        return {}

    df_n = _hf_df[_hf_df['n'] == n]
    if len(df_n) == 0:
        return {}

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

    _cache_data[str(n)] = n_data
    try:
        with open(_cache_path, 'w') as f:
            json.dump(_cache_data, f)
    except Exception:
        pass
    return n_data


def get_baselines_batch(n, m_list):
    n_key = str(n)
    if n_key in _cache_data:
        n_data = _cache_data[n_key]
        result = {int(m): n_data[str(m)] for m in m_list if str(m) in n_data}
        if result:
            return result
    n_data = _load_hf_for_n(n)
    if n_data:
        return {int(m): n_data[str(m)] for m in m_list if str(m) in n_data}
    return {}


def _rescore_affected_np(env, score_fn, logits, n, i, j):
    """Re-score logits for rows/cols of nodes i,j using current RR state. O(N).
    CPU-only version for eval (no device parameter).
    4-dim features: Fiedler gap, deg_i, deg_j, step_frac."""
    v2 = env.V_rr[:, 0]

    nm1 = max(n - 1, 1)
    deg_norm = env.degrees / nm1
    step_frac = env.step / max(env.config.k_steps, 1)

    for nd in (i, j) if i != j else (i,):
        row_feat = np.empty((n, 4), dtype=np.float32)
        row_feat[:, 0] = n * (v2[nd] - v2) ** 2
        row_feat[:, 1] = deg_norm[nd]
        row_feat[:, 2] = deg_norm
        row_feat[:, 3] = step_frac
        with torch.no_grad():
            logits[nd, :] = score_fn(
                torch.tensor(row_feat, dtype=torch.float32)
            ).numpy()

        col_feat = row_feat.copy()
        col_feat[:, 1] = deg_norm
        col_feat[:, 2] = deg_norm[nd]
        with torch.no_grad():
            logits[:, nd] = score_fn(
                torch.tensor(col_feat, dtype=torch.float32)
            ).numpy()


def _eval_single_config(args, policy_state_dict, policy_config_dict,
                        num_trials, k_steps, swap_frac=0.1, stochastic=False,
                        init_method="ring"):
    """Worker: evaluate REFINE agent on a single (n, m) with batch-scored swaps."""
    n, m, baselines = args

    config = RefineConfig(**policy_config_dict)
    policy = RefinePolicy(config)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    num_swaps = max(1, int(m * swap_frac))

    rl_lambda2 = []

    for trial_seed in range(num_trials):
        env_config = RefineEnvConfig(
            min_n=n, max_n=n, k_steps=k_steps, delta=1,
            init_method=init_method)
        env = RefineEnv(env_config, seed=trial_seed)
        env.inference_mode = True
        env.reset(n=n, m=m)

        for k in range(k_steps):
            env.init_rr()

            # === ADD: sequential re-scoring, argmax ===
            edge_feat_add, _ = env.get_edge_level_features_rr()
            ef_add_t = torch.tensor(edge_feat_add, dtype=torch.float32)

            with torch.no_grad():
                add_logits = policy.score_add(ef_add_t).numpy()

            add_mask = (env.adj == 0) & upper_tri
            actual_adds = min(num_swaps, int(add_mask.sum()))
            if actual_adds == 0:
                env.compute_reward()
                continue

            add_logits_live = add_logits.copy()
            add_mask_live = add_mask.copy()
            add_flat_indices = []
            env.begin_add_phase()

            for s in range(actual_adds):
                valid = np.where(add_mask_live.reshape(-1))[0]
                if len(valid) == 0:
                    break
                valid_logits = add_logits_live.reshape(-1)[valid]
                if stochastic:
                    vl = np.clip(valid_logits, -50, 50)
                    vl = vl - vl.max()
                    probs = np.exp(vl)
                    probs = probs / probs.sum()
                    sel = np.random.choice(len(valid), p=probs)
                    flat_idx = int(valid[sel])
                else:
                    flat_idx = int(valid[np.argmax(valid_logits)])
                    if valid_logits[np.argmax(valid_logits)] <= -1e9:
                        break
                add_flat_indices.append(flat_idx)

                ai, aj = flat_idx // n, flat_idx % n
                env.add_single_edge(ai, aj)

                add_mask_live[ai, aj] = False
                _rescore_affected_np(
                    env, policy.score_add, add_logits_live, n, ai, aj)

            num_added = len(add_flat_indices)

            # === REMOVE: sequential re-scoring, argmax ===
            edge_feat_rem, _ = env.get_edge_level_features_rr()
            ef_rem_t = torch.tensor(edge_feat_rem, dtype=torch.float32)

            with torch.no_grad():
                rem_logits = policy.score_remove(ef_rem_t).numpy()

            bridge_mask = env.get_bridge_mask()
            rem_mask = (env.adj > 0) & upper_tri & ~bridge_mask
            for flat_idx in add_flat_indices:
                ai, aj = flat_idx // n, flat_idx % n
                ai2, aj2 = min(ai, aj), max(ai, aj)
                rem_mask[ai2, aj2] = False

            actual_rems = min(num_added, int(rem_mask.sum()))
            if actual_rems == 0:
                # Undo all adds
                for flat_idx in reversed(add_flat_indices):
                    ai, aj = flat_idx // n, flat_idx % n
                    env.adj[ai, aj] = 0
                    env.adj[aj, ai] = 0
                    env.degrees[ai] -= 1
                    env.degrees[aj] -= 1
                    env.m_current -= 1
                    env._edges_added_this_step = 0
                    env.rr_update_remove(ai, aj)
                env.compute_reward()
                continue

            rem_logits_live = rem_logits.copy()
            rem_mask_live = rem_mask.copy()
            rem_flat_indices = []

            for s in range(actual_rems):
                valid = np.where(rem_mask_live.reshape(-1))[0]
                if len(valid) == 0:
                    break
                valid_logits = rem_logits_live.reshape(-1)[valid]
                if stochastic:
                    vl = np.clip(valid_logits, -50, 50)
                    vl = vl - vl.max()
                    probs = np.exp(vl)
                    probs = probs / probs.sum()
                    sel = np.random.choice(len(valid), p=probs)
                    flat_idx = int(valid[sel])
                else:
                    flat_idx = int(valid[np.argmax(valid_logits)])
                    if valid_logits[np.argmax(valid_logits)] <= -1e9:
                        break
                rem_flat_indices.append(flat_idx)

                ri, rj = flat_idx // n, flat_idx % n
                env.adj[ri, rj] = 0
                env.adj[rj, ri] = 0
                env.degrees[ri] -= 1
                env.degrees[rj] -= 1
                env.m_current -= 1
                env._edges_added_this_step = 0
                env.rr_update_remove(ri, rj)

                rem_mask_live[ri, rj] = False
                _rescore_affected_np(
                    env, policy.score_remove, rem_logits_live, n, ri, rj)

            # Undo unmatched adds
            unmatched = num_added - len(rem_flat_indices)
            if unmatched > 0:
                for flat_idx in reversed(add_flat_indices[-unmatched:]):
                    ai, aj = flat_idx // n, flat_idx % n
                    if env.adj[ai, aj] > 0:
                        env.adj[ai, aj] = 0
                        env.adj[aj, ai] = 0
                        env.degrees[ai] -= 1
                        env.degrees[aj] -= 1
                        env.m_current -= 1
                        env._edges_added_this_step = 0
                        env.rr_update_remove(ai, aj)

            env.compute_reward()  # advances step counter

        rl_lambda2.append(algebraic_connectivity(env.adj))

    max_m = n * (n - 1) // 2
    rl_mean = np.mean(rl_lambda2)
    rl_std = np.std(rl_lambda2)

    result = {
        'n': n, 'm': m,
        'density': m / max_m,
        'rl_mean': rl_mean,
        'rl_std': rl_std,
    }

    for key in ['fv', 'er', 'sw_025', 'sw_050', 'sw_075']:
        result[key] = baselines.get(key, None)

    best_baseline = max(
        (baselines.get(k, 0) or 0)
        for k in ['fv', 'er', 'sw_025', 'sw_050', 'sw_075']
    )
    result['best_baseline'] = best_baseline
    result['win'] = (rl_mean > best_baseline + 1e-6
                     if best_baseline > 0 else False)
    # Tie: within 5% of best baseline
    result['tie'] = (not result['win'] and best_baseline > 0
                     and rl_mean >= 0.95 * best_baseline)

    return result


def eval_refine(checkpoint_path, n_values, num_trials=5,
                num_workers=None, k_steps=20, swap_frac=0.1,
                stochastic=False, init_method="ring"):
    """Evaluate REFINE checkpoint against baselines."""
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
        config = checkpoint.get('config', {})
        episode = checkpoint.get('episode', checkpoint.get('epoch', '?'))
        print(f"Checkpoint from episode/epoch {episode}")
    else:
        state_dict = checkpoint
        config = {}

    policy_config_dict = {
        'hidden_dim': config.get('hidden_dim', 64),
        'edge_feat_dim': config.get('edge_feat_dim', 4),
        'graph_feat_dim': config.get('graph_feat_dim', 3),
        'delta': 1,
    }
    print(f"  Policy config: {policy_config_dict}")

    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    n_str = ','.join(str(n) for n in n_values)
    print(f"\nLoading baselines for n={n_str}...")
    tasks = []

    for n in n_values:
        max_m = n * (n - 1) // 2
        min_m = n - 1
        all_m = list(range(min_m, max_m + 1))
        batch = get_baselines_batch(n, all_m)
        if not batch:
            print(f"  n={n}: no baselines found, skipping")
            continue
        for m_val in all_m:
            bl = batch.get(m_val, {})
            if bl:
                tasks.append((n, m_val, bl))

    total = len(tasks)
    print(f"  Total: {total} configs, {num_trials} trials each, "
          f"{num_workers} workers")

    worker_fn = partial(
        _eval_single_config,
        policy_state_dict=state_dict,
        policy_config_dict=policy_config_dict,
        num_trials=num_trials,
        k_steps=k_steps,
        swap_frac=swap_frac,
        stochastic=stochastic,
        init_method=init_method,
    )

    results = []
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, tasks)):
            results.append(result)
            if (i + 1) % 10 == 0 or i + 1 == total:
                print(f"  {i+1}/{total} configs evaluated...", flush=True)

    results.sort(key=lambda r: (r['n'], r['m']))

    def _v(val):
        return f"{val:>9.4f}" if val is not None and val > 0 else f"{'---':>9}"

    wins = sum(1 for r in results if r['win'])
    ties = sum(1 for r in results if r['tie'])

    print(f"\n{'='*115}")
    print(f"RESULTS: REFINE Agent vs ALL Baselines ({num_trials} trials/config)")
    print(f"{'='*115}")

    hdr = (f"{'(n,m)':>9}  {'FV':>9} {'ER':>9} {'SW.25':>9} "
           f"{'SW.50':>9} {'SW.75':>9} {'RL':>9} {'':>5}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for r in results:
        fv = r.get('fv') or 0
        er = r.get('er') or 0
        s25 = r.get('sw_025') or 0
        s50 = r.get('sw_050') or 0
        s75 = r.get('sw_075') or 0
        rl = r['rl_mean']
        if r['win']:
            tag = "  WIN"
        elif r['tie']:
            tag = "  TIE"
        else:
            tag = ""
        label = f"({r['n']},{r['m']})"
        print(f"{label:>9}  {_v(fv)} {_v(er)} {_v(s25)} "
              f"{_v(s50)} {_v(s75)} {rl:>9.4f} {tag}")

    print("-" * 115)

    print(f"\nSUMMARY: {wins} WINS, {ties} TIES / {total} configs")

    with_baseline = sum(1 for r in results if r['best_baseline'] > 0)
    reach_95 = sum(1 for r in results
                   if r['best_baseline'] > 0
                   and r['rl_mean'] >= 0.95 * r['best_baseline'])
    reach_99 = sum(1 for r in results
                   if r['best_baseline'] > 0
                   and r['rl_mean'] >= 0.99 * r['best_baseline'])
    print(f"  >= 99% of best baseline: {reach_99}/{with_baseline}")
    print(f"  >= 95% of best baseline: {reach_95}/{with_baseline}")

    for n in sorted(set(r['n'] for r in results)):
        nr = [r for r in results if r['n'] == n]
        nw = sum(1 for r in nr if r['win'])
        nt = sum(1 for r in nr if r['tie'])
        avg_rl = np.mean([r['rl_mean'] for r in nr])
        avg_best = np.mean([r['best_baseline'] for r in nr
                           if r['best_baseline'] > 0])
        print(f"  n={n}: {nw} wins, {nt} ties / {len(nr)}, "
              f"avg RL={avg_rl:.3f}, avg best={avg_best:.3f}")

    print(f"{'='*115}")

    csv_path = (Path(checkpoint_path).parent
                / f"eval_refine_n{n_str.replace(',', '_')}.csv")
    with open(csv_path, 'w') as f:
        f.write("n,m,density,rl,rl_std,fv,er,sw025,sw050,sw075,win,tie\n")
        for r in results:
            f.write(f"{r['n']},{r['m']},{r['density']:.4f},"
                    f"{r['rl_mean']:.6f},{r['rl_std']:.6f},"
                    f"{r.get('fv','')},{r.get('er','')},"
                    f"{r.get('sw_025','')},{r.get('sw_050','')},"
                    f"{r.get('sw_075','')},"
                    f"{1 if r['win'] else 0},"
                    f"{1 if r['tie'] else 0}\n")
    print(f"\nCSV saved: {csv_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate REFINE checkpoint")
    parser.add_argument("checkpoint",
                        help="Path to REFINE checkpoint file")
    parser.add_argument("--n", type=str, required=True,
                        help="Comma-separated n values (e.g. 8,10,12,14,16)")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k-steps", type=int, default=20)
    parser.add_argument("--swap-frac", type=float, default=0.1,
                        help="Fraction of edges to swap per step (default 0.1)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use softmax sampling instead of argmax")
    parser.add_argument("--init", type=str, default="ring",
                        choices=["ring", "fv"],
                        help="Graph init: ring (ring+random) or fv (FV greedy)")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]
    eval_refine(args.checkpoint, n_values=n_values,
                num_trials=args.trials, num_workers=args.workers,
                k_steps=args.k_steps, swap_frac=args.swap_frac,
                stochastic=args.stochastic, init_method=args.init)
