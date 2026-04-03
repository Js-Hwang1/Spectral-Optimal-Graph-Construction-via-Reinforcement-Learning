#!/usr/bin/env python3
"""
PoC: How much does re-scoring ONLY the affected rows (nodes involved in swap)
recover vs full O(N²) re-scoring after each swap?

Compares three strategies after each swap:
  1. STALE:  no re-scoring (batch v7 — use initial scores for all swaps)
  2. ROW:    re-score only the 2 rows of swapped nodes (O(N) update)
  3. FRESH:  re-score all N² pairs (regular v7 — O(N²) update)

Measures: how often does each strategy pick the same "best next swap"
as FRESH? And how close is the score of their pick to the FRESH best?

Runs on actual REFINE env with trained policy from v7 checkpoint.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.refine_env import RefineEnv, RefineEnvConfig
from models.refine_policy import RefinePolicy, RefineConfig
from utils.spectral import exact_lambda2_np


def score_all_pairs(policy, env, n, upper_tri, phase='add'):
    """Score all N² pairs. Returns (N,N) logits and (N,N,6) features."""
    edge_feat, _ = env.get_edge_level_features_rr()
    ef_t = torch.tensor(edge_feat, dtype=torch.float32)
    with torch.no_grad():
        if phase == 'add':
            logits = policy.score_add(ef_t)
        else:
            logits = policy.score_remove(ef_t)
    return logits.numpy(), edge_feat


def score_rows(policy, env, nodes, n, phase='add'):
    """Score only specific rows. Returns dict: node_i -> (N,) logits, (N,6) feats."""
    results = {}
    for i in nodes:
        row_feat, _ = env.get_row_features_rr(i)
        rf_t = torch.tensor(row_feat, dtype=torch.float32)
        with torch.no_grad():
            if phase == 'add':
                row_logits = policy.score_add(rf_t)
            else:
                row_logits = policy.score_remove(rf_t)
        results[i] = (row_logits.numpy(), row_feat)
    return results


def get_best_add(logits, adj, upper_tri):
    """Find best non-edge from logits."""
    mask = (adj == 0) & upper_tri
    if not mask.any():
        return None, -1e9
    masked = np.where(mask, logits, -1e9)
    flat = masked.reshape(-1)
    idx = np.argmax(flat)
    n = logits.shape[0]
    return (idx // n, idx % n), flat[idx]


def run_poc(checkpoint_path, n_values=[8, 16, 32], num_swaps_list=None,
            num_trials=20, k_step_idx=0):
    """Run the PoC comparison."""

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['policy_state_dict']
    config = ckpt.get('config', {})

    policy_config = RefineConfig(
        hidden_dim=config.get('hidden_dim', 64),
        edge_feat_dim=config.get('edge_feat_dim', 6),
        graph_feat_dim=config.get('graph_feat_dim', 5),
        delta=1,
    )
    policy = RefinePolicy(policy_config)
    policy.load_state_dict(state_dict)
    policy.eval()

    print("=" * 80)
    print("PoC: Row-Update vs Stale vs Fresh MLP Re-scoring")
    print("=" * 80)

    for n in n_values:
        if num_swaps_list is None:
            num_swaps = max(1, n // 8)
        else:
            num_swaps = num_swaps_list[n_values.index(n)]

        upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

        # Track agreement with FRESH across all trials
        stale_agrees = []      # does STALE pick same edge as FRESH?
        row_agrees = []        # does ROW pick same edge as FRESH?
        stale_score_ratios = []  # score(stale_pick) / score(fresh_pick) under fresh scoring
        row_score_ratios = []

        # Track actual lambda2 improvements
        stale_lambda2s = []
        row_lambda2s = []
        fresh_lambda2s = []

        for trial in range(num_trials):
            m = n + n // 2  # moderate density
            env_config = RefineEnvConfig(min_n=n, max_n=n, k_steps=20, delta=1)

            # We'll simulate: do 1 swap with fresh, then compare strategies for swap #2
            for swap_idx in range(1, num_swaps):
                # Fresh env for each strategy
                env_fresh = RefineEnv(env_config, seed=trial * 100 + swap_idx)
                env_fresh.reset(n=n, m=m)
                env_fresh.init_rr()

                # Do swap #1 (same for all strategies)
                logits_0, feat_0 = score_all_pairs(policy, env_fresh, n, upper_tri, 'add')
                add_mask = (env_fresh.adj == 0) & upper_tri
                best_add, _ = get_best_add(logits_0, env_fresh.adj, upper_tri)
                if best_add is None:
                    continue
                ai, aj = best_add

                env_fresh.begin_add_phase()
                env_fresh.add_single_edge(ai, aj)

                # Now score REMOVE, pick best, do it
                rem_logits_0, _ = score_all_pairs(policy, env_fresh, n, upper_tri, 'remove')
                bridge_mask = env_fresh.get_bridge_mask()
                rem_mask = (env_fresh.adj > 0) & upper_tri & ~bridge_mask
                ai2, aj2 = min(ai, aj), max(ai, aj)
                rem_mask[ai2, aj2] = False
                if not rem_mask.any():
                    continue
                masked_rem = np.where(rem_mask, rem_logits_0, -1e9)
                rem_flat = np.argmax(masked_rem.reshape(-1))
                ri, rj = rem_flat // n, rem_flat % n

                # Apply remove to fresh env
                env_fresh.adj[ri, rj] = 0
                env_fresh.adj[rj, ri] = 0
                env_fresh.degrees[ri] -= 1
                env_fresh.degrees[rj] -= 1
                env_fresh.m_current -= 1
                env_fresh.rr_update_remove(ri, rj)

                # Swap #1 done. Nodes involved: ai, aj (add), ri, rj (remove)
                involved_nodes = list(set([ai, aj, ri, rj]))

                # === Now compare three strategies for swap #2 ADD ===

                # Strategy 1: FRESH — re-score everything
                fresh_logits, _ = score_all_pairs(
                    policy, env_fresh, n, upper_tri, 'add')
                add_mask_2 = (env_fresh.adj == 0) & upper_tri
                fresh_best, fresh_score = get_best_add(
                    fresh_logits, env_fresh.adj, upper_tri)

                if fresh_best is None:
                    continue

                # Strategy 2: STALE — use the ORIGINAL logits (from before swap #1)
                # But update the mask (we know which edges changed)
                stale_logits = logits_0.copy()
                stale_best, stale_score = get_best_add(
                    stale_logits, env_fresh.adj, upper_tri)

                # Strategy 3: ROW — update only involved rows
                row_logits = logits_0.copy()
                row_updates = score_rows(
                    policy, env_fresh, involved_nodes, n, 'add')
                for node_i, (row_log, _) in row_updates.items():
                    row_logits[node_i, :] = row_log
                    # Also update the column (symmetric scoring)
                    # Actually, our features are asymmetric (deg_i vs deg_j differ)
                    # So we need to update columns too for pairs (j, node_i)
                    # But upper_tri means we only look at (i,j) where i<j
                    # So updating row node_i covers pairs (node_i, j>node_i)
                    # We also need (j, node_i) for j < node_i
                    # Those are in row j, column node_i — need to re-score row j?
                    # No — we can re-score the COLUMN of node_i
                    # by getting features for pairs (j, node_i) for all j
                    # Actually, get_row_features_rr(node_i) gives features
                    # for (node_i, j), not (j, node_i). The features differ
                    # because deg_i and deg_j are swapped.
                    # For upper_tri pairs (j, node_i) where j < node_i,
                    # we need features with j as source and node_i as target.
                    pass

                # Actually need to handle columns properly.
                # For each involved node k, update:
                #   row k: pairs (k, j) for j > k  [in upper_tri]
                #   col k: pairs (j, k) for j < k  [also in upper_tri]
                # get_row_features_rr(k) gives (k, j) features.
                # For (j, k) features, we need get_row_features_rr(j)... O(N²) total
                #
                # SIMPLER: just re-build features for the involved rows AND columns.
                # For column of node k: features (j, k) have deg_j as feat[2], deg_k as feat[3]
                # We can construct these from the RR state directly.

                # Let's do it properly: for each involved node k,
                # update row k AND column k in the logits matrix.
                row_logits = logits_0.copy()
                v2, v3, lam2, lam3 = env_fresh.get_rr_spectral_info()
                nm1 = max(n - 1, 1)
                deg_norm = env_fresh.degrees / nm1
                step_frac = env_fresh.step / max(env_fresh.config.k_steps, 1)
                spectral_gap = max(lam3 - lam2, 0.0) / max(n, 1)

                for k in involved_nodes:
                    # Row k: features for (k, j)
                    row_feat = np.zeros((n, 6), dtype=np.float32)
                    row_feat[:, 0] = n * (v2[k] - v2[:]) ** 2
                    row_feat[:, 1] = n * (v3[k] - v3[:]) ** 2
                    row_feat[:, 2] = deg_norm[k]
                    row_feat[:, 3] = deg_norm[:]
                    row_feat[:, 4] = spectral_gap
                    row_feat[:, 5] = step_frac
                    rf_t = torch.tensor(row_feat, dtype=torch.float32)
                    with torch.no_grad():
                        row_logits[k, :] = policy.score_add(rf_t).numpy()

                    # Col k: features for (j, k) where j < k
                    col_feat = np.zeros((n, 6), dtype=np.float32)
                    col_feat[:, 0] = n * (v2[:] - v2[k]) ** 2  # same as row
                    col_feat[:, 1] = n * (v3[:] - v3[k]) ** 2  # same as row
                    col_feat[:, 2] = deg_norm[:]       # j's degree
                    col_feat[:, 3] = deg_norm[k]       # k's degree
                    col_feat[:, 4] = spectral_gap
                    col_feat[:, 5] = step_frac
                    cf_t = torch.tensor(col_feat, dtype=torch.float32)
                    with torch.no_grad():
                        row_logits[:, k] = policy.score_add(cf_t).numpy()

                row_best, row_score_val = get_best_add(
                    row_logits, env_fresh.adj, upper_tri)

                # Compare
                stale_match = (stale_best == fresh_best) if stale_best else False
                row_match = (row_best == fresh_best) if row_best else False

                stale_agrees.append(1 if stale_match else 0)
                row_agrees.append(1 if row_match else 0)

                # Score of each pick under FRESH scoring
                if fresh_best and stale_best:
                    stale_pick_fresh_score = fresh_logits[stale_best[0], stale_best[1]]
                    ratio = stale_pick_fresh_score / (fresh_score + 1e-10)
                    stale_score_ratios.append(ratio)

                if fresh_best and row_best:
                    row_pick_fresh_score = fresh_logits[row_best[0], row_best[1]]
                    ratio = row_pick_fresh_score / (fresh_score + 1e-10)
                    row_score_ratios.append(ratio)

                # === Measure actual lambda2 impact ===
                # Apply each strategy's pick and measure lambda2

                # Fresh pick
                adj_fresh = env_fresh.adj.copy()
                fi, fj = fresh_best
                adj_fresh[fi, fj] = 1
                adj_fresh[fj, fi] = 1
                fresh_lambda2s.append(exact_lambda2_np(adj_fresh))

                # Stale pick
                if stale_best:
                    adj_stale = env_fresh.adj.copy()
                    si, sj = stale_best
                    if adj_stale[si, sj] == 0:  # valid non-edge
                        adj_stale[si, sj] = 1
                        adj_stale[sj, si] = 1
                        stale_lambda2s.append(exact_lambda2_np(adj_stale))

                # Row pick
                if row_best:
                    adj_row = env_fresh.adj.copy()
                    oi, oj = row_best
                    if adj_row[oi, oj] == 0:
                        adj_row[oi, oj] = 1
                        adj_row[oj, oi] = 1
                        row_lambda2s.append(exact_lambda2_np(adj_row))

        # Report
        print(f"\n--- n={n}, num_swaps={num_swaps}, "
              f"{len(stale_agrees)} comparisons ---")

        if stale_agrees:
            print(f"  STALE agreement with FRESH: "
                  f"{np.mean(stale_agrees)*100:.1f}%")
            print(f"  ROW   agreement with FRESH: "
                  f"{np.mean(row_agrees)*100:.1f}%")

        if stale_score_ratios:
            print(f"  STALE score ratio (pick quality): "
                  f"{np.mean(stale_score_ratios):.4f}")
            print(f"  ROW   score ratio (pick quality): "
                  f"{np.mean(row_score_ratios):.4f}")

        if fresh_lambda2s:
            print(f"  lambda2 after FRESH pick: {np.mean(fresh_lambda2s):.4f}")
        if stale_lambda2s:
            print(f"  lambda2 after STALE pick: {np.mean(stale_lambda2s):.4f}")
        if row_lambda2s:
            print(f"  lambda2 after ROW   pick: {np.mean(row_lambda2s):.4f}")

        if fresh_lambda2s and stale_lambda2s and row_lambda2s:
            min_len = min(len(fresh_lambda2s), len(stale_lambda2s),
                          len(row_lambda2s))
            fresh_arr = np.array(fresh_lambda2s[:min_len])
            stale_arr = np.array(stale_lambda2s[:min_len])
            row_arr = np.array(row_lambda2s[:min_len])

            stale_gap = np.mean(fresh_arr - stale_arr)
            row_gap = np.mean(fresh_arr - row_arr)
            print(f"  lambda2 gap FRESH-STALE: {stale_gap:+.4f}")
            print(f"  lambda2 gap FRESH-ROW:   {row_gap:+.4f}")
            if abs(stale_gap) > 1e-8:
                recovery = 1.0 - row_gap / stale_gap
                print(f"  ROW recovers {recovery*100:.1f}% of the gap")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint")
    parser.add_argument("--n", type=str, default="8,16,32",
                        help="Comma-separated n values")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n.split(',')]
    run_poc(args.checkpoint, n_values=n_values, num_trials=args.trials)
