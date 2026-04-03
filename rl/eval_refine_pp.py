#!/usr/bin/env python3
"""
Evaluate REFINE PP (v9) checkpoint against baselines.

Flat 3M swap loop evaluation:
  - 3*m total swaps per episode (flat, no K-step nesting)
  - Periodic Lanczos refresh every max(1, m//10) swaps
  - Deterministic: argmax instead of sampling
  - No eigensolves at inference

Usage:
    python eval_refine_pp.py checkpoints/best.pt --n 8 --trials 5
    python eval_refine_pp.py checkpoints/best.pt --n 8,10,12,16,24,32,36 --trials 10
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


# ============================================================================
# O(N) row+col update of logits for involved nodes (deterministic)
# ============================================================================

def _update_logits_for_nodes(policy, env, logits, nodes, n, phase='add'):
    """Re-score rows and columns of `nodes` in the (N,N) logits matrix."""
    v2, v3, lam2, lam3 = env.get_rr_spectral_info()
    v4 = env.V_rr[:, 2] if env.V_rr.shape[1] > 2 else np.zeros(n)
    v5 = env.V_rr[:, 3] if env.V_rr.shape[1] > 3 else np.zeros(n)
    lam4 = float(env.lams_rr[2]) if len(env.lams_rr) > 2 else lam3
    lam5 = float(env.lams_rr[3]) if len(env.lams_rr) > 3 else lam4

    nm1 = max(n - 1, 1)
    deg_norm = env.degrees / nm1
    step_frac = env.step / max(env.config.k_steps, 1)
    gap_23 = max(lam3 - lam2, 0.0) / max(n, 1)
    gap_34 = max(lam4 - lam3, 0.0) / max(n, 1)
    gap_45 = max(lam5 - lam4, 0.0) / max(n, 1)

    score_fn = policy.score_add if phase == 'add' else policy.score_remove

    for k in nodes:
        row_feat = np.zeros((n, 10), dtype=np.float32)
        row_feat[:, 0] = n * (v2[k] - v2[:]) ** 2
        row_feat[:, 1] = n * (v3[k] - v3[:]) ** 2
        row_feat[:, 2] = n * (v4[k] - v4[:]) ** 2
        row_feat[:, 3] = n * (v5[k] - v5[:]) ** 2
        row_feat[:, 4] = deg_norm[k]
        row_feat[:, 5] = deg_norm[:]
        row_feat[:, 6] = gap_23
        row_feat[:, 7] = gap_34
        row_feat[:, 8] = gap_45
        row_feat[:, 9] = step_frac
        rf_t = torch.tensor(row_feat, dtype=torch.float32)
        with torch.no_grad():
            logits[k, :] = score_fn(rf_t).numpy()

        col_feat = np.zeros((n, 10), dtype=np.float32)
        col_feat[:, 0] = n * (v2[:] - v2[k]) ** 2
        col_feat[:, 1] = n * (v3[:] - v3[k]) ** 2
        col_feat[:, 2] = n * (v4[:] - v4[k]) ** 2
        col_feat[:, 3] = n * (v5[:] - v5[k]) ** 2
        col_feat[:, 4] = deg_norm[:]
        col_feat[:, 5] = deg_norm[k]
        col_feat[:, 6] = gap_23
        col_feat[:, 7] = gap_34
        col_feat[:, 8] = gap_45
        col_feat[:, 9] = step_frac
        cf_t = torch.tensor(col_feat, dtype=torch.float32)
        with torch.no_grad():
            logits[:, k] = score_fn(cf_t).numpy()


# ============================================================================
# Worker: flat 3M swap evaluation
# ============================================================================

def _eval_single_config(args, policy_state_dict, policy_config_dict,
                        num_trials):
    """Worker: evaluate REFINE PP agent on a single (n, m) with flat 3M swap loop."""
    n, m, baselines = args

    config = RefineConfig(**policy_config_dict)
    policy = RefinePolicy(config)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    total_swaps = 3 * m
    refresh_interval = max(1, m // 10)

    rl_lambda2 = []

    for trial_seed in range(num_trials):
        env_config = RefineEnvConfig(
            min_n=n, max_n=n, k_steps=total_swaps, delta=1)
        env = RefineEnv(env_config, seed=trial_seed)
        env.inference_mode = True
        env.reset(n=n, m=m)

        # Initial Lanczos + batch score
        env.init_rr()
        edge_feat, _ = env.get_edge_level_features_rr()
        ef_t = torch.tensor(edge_feat, dtype=torch.float32)
        with torch.no_grad():
            add_logits = policy.score_add(ef_t).numpy()
            rem_logits = policy.score_remove(ef_t).numpy()

        for i in range(total_swaps):
            env.step = i

            # Periodic refresh
            if i > 0 and i % refresh_interval == 0:
                env.init_rr()
                edge_feat, _ = env.get_edge_level_features_rr()
                ef_t = torch.tensor(edge_feat, dtype=torch.float32)
                with torch.no_grad():
                    add_logits = policy.score_add(ef_t).numpy()
                    rem_logits = policy.score_remove(ef_t).numpy()

            # --- ADD: argmax from add_logits ---
            add_mask = (env.adj == 0) & upper_tri
            if not add_mask.any():
                continue

            masked_add = np.where(add_mask, add_logits, -1e9)
            add_flat_idx = int(np.argmax(masked_add.reshape(-1)))
            ai, aj = add_flat_idx // n, add_flat_idx % n

            env.begin_add_phase()
            env.add_single_edge(ai, aj)

            # Row-update logits
            involved_add = list(set([ai, aj]))
            _update_logits_for_nodes(
                policy, env, add_logits, involved_add, n, 'add')
            _update_logits_for_nodes(
                policy, env, rem_logits, involved_add, n, 'remove')

            # --- REMOVE: argmax from rem_logits ---
            bridge_mask = env.get_bridge_mask()
            rem_mask = (env.adj > 0) & upper_tri & ~bridge_mask
            ai2, aj2 = min(ai, aj), max(ai, aj)
            rem_mask[ai2, aj2] = False  # exclude just-added

            if not rem_mask.any():
                # Undo add and continue
                env.adj[ai, aj] = 0
                env.adj[aj, ai] = 0
                env.degrees[ai] -= 1
                env.degrees[aj] -= 1
                env.m_current -= 1
                env._edges_added_this_step = 0
                env.rr_update_remove(ai, aj)
                _update_logits_for_nodes(
                    policy, env, add_logits, involved_add, n, 'add')
                _update_logits_for_nodes(
                    policy, env, rem_logits, involved_add, n, 'remove')
                continue

            masked_rem = np.where(rem_mask, rem_logits, -1e9)
            rem_flat_idx = int(np.argmax(masked_rem.reshape(-1)))
            ri, rj = rem_flat_idx // n, rem_flat_idx % n

            # Remove edge + RR update
            env.adj[ri, rj] = 0
            env.adj[rj, ri] = 0
            env.degrees[ri] -= 1
            env.degrees[rj] -= 1
            env.m_current -= 1
            env._edges_added_this_step = 0
            env.rr_update_remove(ri, rj)

            # Row-update for remove
            involved_rem = list(set([ri, rj]))
            _update_logits_for_nodes(
                policy, env, add_logits, involved_rem, n, 'add')
            _update_logits_for_nodes(
                policy, env, rem_logits, involved_rem, n, 'remove')

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
    result['tie'] = (not result['win'] and best_baseline > 0
                     and rl_mean >= 0.95 * best_baseline)

    return result


def eval_refine_pp(checkpoint_path, n_values, num_trials=5,
                   num_workers=None):
    """Evaluate REFINE PP checkpoint against baselines."""
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
        'edge_feat_dim': config.get('edge_feat_dim', 6),
        'graph_feat_dim': config.get('graph_feat_dim', 5),
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
    print(f"RESULTS: REFINE PP Agent vs ALL Baselines ({num_trials} trials/config)")
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
                / f"eval_refine_pp_n{n_str.replace(',', '_')}.csv")
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
        description="Evaluate REFINE PP (v9) checkpoint")
    parser.add_argument("checkpoint",
                        help="Path to REFINE checkpoint file")
    parser.add_argument("--n", type=str, required=True,
                        help="Comma-separated n values (e.g. 8,10,12,16,24,32)")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]
    eval_refine_pp(args.checkpoint, n_values=n_values,
                   num_trials=args.trials, num_workers=args.workers)
