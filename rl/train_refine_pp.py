#!/usr/bin/env python3
"""
REFINE v9 (PP) — Flat 3M Swap Loop with Per-Swap Rewards.

Replaces v8's nested K-step × M/10 structure with a single flat loop of 3M swaps.
Expensive Lanczos + batch-score done upfront, then O(N) cheap updates per swap.
Periodic Lanczos refresh every max(1, m//10) swaps (~30 refreshes total).

Training: O(N³) eigensolve per swap (acceptable at n=8-10).
Inference: Same batch-scored + row-update structure, no eigensolves.

Usage:
    python train_refine_pp.py --min-n 8 --max-n 10 --episodes 20000
"""

import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from envs.refine_env import RefineEnv, RefineEnvConfig
from envs.gnm_env import algebraic_connectivity
from models.refine_policy import RefinePolicy, RefineConfig
from utils.spectral import exact_lambda2_np


# ============================================================================
# O(N) row+col update of logits for involved nodes
# ============================================================================

def update_logits_for_nodes(policy, env, logits, nodes, n, phase='add'):
    """
    Re-score rows and columns of `nodes` in the (N,N) logits matrix.
    Uses current RR spectral state. O(N) per node, O(N * |nodes|) total.
    """
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
        # Row k: features for (k, j) for all j
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

        # Col k: features for (j, k) for all j
        col_feat = np.zeros((n, 10), dtype=np.float32)
        col_feat[:, 0] = n * (v2[:] - v2[k]) ** 2
        col_feat[:, 1] = n * (v3[:] - v3[k]) ** 2
        col_feat[:, 2] = n * (v4[:] - v4[k]) ** 2
        col_feat[:, 3] = n * (v5[:] - v5[k]) ** 2
        col_feat[:, 4] = deg_norm[:]       # j's degree as source
        col_feat[:, 5] = deg_norm[k]       # k's degree as target
        col_feat[:, 6] = gap_23
        col_feat[:, 7] = gap_34
        col_feat[:, 8] = gap_45
        col_feat[:, 9] = step_frac
        cf_t = torch.tensor(col_feat, dtype=torch.float32)
        with torch.no_grad():
            logits[:, k] = score_fn(cf_t).numpy()


# ============================================================================
# Helper: graph-level features from RR state (O(N))
# ============================================================================

def get_graph_feat_rr(env):
    """Quick graph-level features from current RR state."""
    v2, v3, lam2, lam3 = env.get_rr_spectral_info()
    lam4 = float(env.lams_rr[2]) if len(env.lams_rr) > 2 else lam3
    lam5 = float(env.lams_rr[3]) if len(env.lams_rr) > 3 else lam4
    n = env.n
    nm1 = max(n - 1, 1)
    deg_mean = (env.degrees / nm1).mean()
    step_frac = env.step / max(env.config.k_steps, 1)
    gap_23 = max(lam3 - lam2, 0.0) / max(n, 1)
    gap_34 = max(lam4 - lam3, 0.0) / max(n, 1)
    gap_45 = max(lam5 - lam4, 0.0) / max(n, 1)
    return np.array([
        deg_mean, lam2 / max(n, 1), lam3 / max(n, 1),
        lam4 / max(n, 1), lam5 / max(n, 1),
        gap_23, gap_34, gap_45, step_frac
    ], dtype=np.float32)


# ============================================================================
# Build (N,N,10) features from current RR state
# ============================================================================

def self_build_features(env, n):
    """Build (N,N,10) features from current env RR state. O(N²)."""
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

    feat = np.zeros((n, n, 10), dtype=np.float32)
    feat[:, :, 0] = n * (v2[:, None] - v2[None, :]) ** 2
    feat[:, :, 1] = n * (v3[:, None] - v3[None, :]) ** 2
    feat[:, :, 2] = n * (v4[:, None] - v4[None, :]) ** 2
    feat[:, :, 3] = n * (v5[:, None] - v5[None, :]) ** 2
    feat[:, :, 4] = deg_norm[:, None]
    feat[:, :, 5] = deg_norm[None, :]
    feat[:, :, 6] = gap_23
    feat[:, :, 7] = gap_34
    feat[:, :, 8] = gap_45
    feat[:, :, 9] = step_frac
    return feat


# ============================================================================
# PPO re-evaluation (per-swap)
# ============================================================================

def evaluate_swap(policy, ctx, device):
    """
    Re-evaluate one swap under the current policy (differentiable).
    Returns (log_prob, entropy, value) for PPO update.
    """
    n = ctx['n']
    gf_t = torch.tensor(ctx['graph_feat'], dtype=torch.float32, device=device)
    value = policy.compute_value(gf_t)

    lps = []
    total_entropy = torch.tensor(0.0, device=device)

    # Re-evaluate ADD
    ef_add_t = torch.tensor(
        ctx['add_feat'], dtype=torch.float32, device=device)
    add_logits = policy.score_add(ef_add_t).view(-1)

    add_mask_t = torch.tensor(
        ctx['add_mask'], dtype=torch.bool, device=device).view(-1)
    valid_add = torch.where(add_mask_t)[0]

    if len(valid_add) > 0:
        valid_logits = add_logits[valid_add]
        lp = F.log_softmax(valid_logits, dim=0)
        p = torch.exp(lp)
        total_entropy = total_entropy - (p * lp).sum()

        idx_to_pos = torch.full((n * n,), -1, dtype=torch.long,
                                device=device)
        idx_to_pos[valid_add] = torch.arange(
            len(valid_add), device=device)
        pos = idx_to_pos[ctx['add_flat_idx']]
        if pos >= 0:
            lps.append(lp[pos])

    # Re-evaluate REMOVE
    ef_rem_t = torch.tensor(
        ctx['rem_feat'], dtype=torch.float32, device=device)
    rem_logits = policy.score_remove(ef_rem_t).view(-1)

    rem_mask_t = torch.tensor(
        ctx['rem_mask'], dtype=torch.bool, device=device).view(-1)
    valid_rem = torch.where(rem_mask_t)[0]

    if len(valid_rem) > 0:
        valid_logits = rem_logits[valid_rem]
        lp = F.log_softmax(valid_logits, dim=0)
        p = torch.exp(lp)
        total_entropy = total_entropy - (p * lp).sum()

        idx_to_pos = torch.full((n * n,), -1, dtype=torch.long,
                                device=device)
        idx_to_pos[valid_rem] = torch.arange(
            len(valid_rem), device=device)
        pos = idx_to_pos[ctx['rem_flat_idx']]
        if pos >= 0:
            lps.append(lp[pos])

    if lps:
        total_lp = torch.stack(lps).sum()
    else:
        total_lp = torch.tensor(0.0, device=device, requires_grad=True)

    return total_lp, total_entropy, value


# ============================================================================
# Episode collection: FLAT 3M swap loop with per-swap rewards
# ============================================================================

def collect_episode_pp(env, policy, n, m, device):
    """
    Collect one episode with a flat loop of 3*m paired swaps.

    No K-step nesting. Periodic Lanczos refresh every max(1, m//10) swaps.
    Each swap (add + remove) gets its own exact Δλ₂ reward.
    """
    total_swaps = 3 * m
    refresh_interval = max(1, m // 10)

    # Configure env so step_frac = i / total_swaps
    env.config.k_steps = total_swaps
    env.reset(n=n, m=m)

    swap_values = []
    swap_log_probs = []
    swap_rewards = []
    swap_entropies = []
    swap_ctxs = []

    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    lambda2_cache = env.lambda2  # from reset(), exact

    # Initial Lanczos + batch score [O(N²)]
    env.init_rr()
    edge_feat, _ = env.get_edge_level_features_rr()
    ef_t = torch.tensor(edge_feat, dtype=torch.float32, device=device)
    with torch.no_grad():
        add_logits = policy.score_add(ef_t).cpu().numpy()
        rem_logits = policy.score_remove(ef_t).cpu().numpy()

    for i in range(total_swaps):
        env.step = i  # step_frac = i / total_swaps

        # Periodic refresh
        if i > 0 and i % refresh_interval == 0:
            env.init_rr()
            edge_feat, _ = env.get_edge_level_features_rr()
            ef_t = torch.tensor(edge_feat, dtype=torch.float32, device=device)
            with torch.no_grad():
                add_logits = policy.score_add(ef_t).cpu().numpy()
                rem_logits = policy.score_remove(ef_t).cpu().numpy()

        # --- Value for this swap's state ---
        gf = get_graph_feat_rr(env)
        gf_t = torch.tensor(gf, dtype=torch.float32, device=device)
        with torch.no_grad():
            value = policy.compute_value(gf_t)

        # --- ADD: sample from current add_logits ---
        add_mask = (env.adj == 0) & upper_tri
        add_mask_flat = add_mask.reshape(-1)
        valid_add = np.where(add_mask_flat)[0]

        if len(valid_add) == 0:
            continue

        # Store features for PPO replay
        add_feat = self_build_features(env, n)

        valid_logits_t = torch.tensor(
            add_logits.reshape(-1)[valid_add], dtype=torch.float32)
        add_lp = F.log_softmax(valid_logits_t, dim=0)
        add_p = torch.exp(add_lp)

        sel = torch.multinomial(add_p, 1).squeeze()
        add_flat_idx = int(valid_add[sel])
        ai, aj = add_flat_idx // n, add_flat_idx % n
        add_log_prob = add_lp[sel].item()
        add_ent = -(add_p * add_lp).sum().item()

        # Add edge + RR update
        env.begin_add_phase()
        env.add_single_edge(ai, aj)

        # Row-update logits for involved nodes O(N)
        involved_add = list(set([ai, aj]))
        update_logits_for_nodes(
            policy, env, add_logits, involved_add, n, 'add')
        update_logits_for_nodes(
            policy, env, rem_logits, involved_add, n, 'remove')

        # --- REMOVE: sample from updated rem_logits ---
        bridge_mask = env.get_bridge_mask()
        rem_mask = (env.adj > 0) & upper_tri & ~bridge_mask
        ai2, aj2 = min(ai, aj), max(ai, aj)
        rem_mask[ai2, aj2] = False  # exclude just-added

        rem_mask_flat = rem_mask.reshape(-1)
        valid_rem = np.where(rem_mask_flat)[0]

        if len(valid_rem) == 0:
            # Undo add and continue to next swap
            env.adj[ai, aj] = 0
            env.adj[aj, ai] = 0
            env.degrees[ai] -= 1
            env.degrees[aj] -= 1
            env.m_current -= 1
            env._edges_added_this_step = 0
            env.rr_update_remove(ai, aj)
            # Revert logits
            update_logits_for_nodes(
                policy, env, add_logits, involved_add, n, 'add')
            update_logits_for_nodes(
                policy, env, rem_logits, involved_add, n, 'remove')
            continue

        # Store features for PPO replay (post-add state)
        rem_feat = self_build_features(env, n)

        valid_rem_logits_t = torch.tensor(
            rem_logits.reshape(-1)[valid_rem], dtype=torch.float32)
        rem_lp = F.log_softmax(valid_rem_logits_t, dim=0)
        rem_p = torch.exp(rem_lp)

        rem_sel = torch.multinomial(rem_p, 1).squeeze()
        rem_flat_idx = int(valid_rem[rem_sel])
        ri, rj = rem_flat_idx // n, rem_flat_idx % n
        rem_log_prob = rem_lp[rem_sel].item()
        rem_ent = -(rem_p * rem_lp).sum().item()

        # Remove edge + RR update
        env.adj[ri, rj] = 0
        env.adj[rj, ri] = 0
        env.degrees[ri] -= 1
        env.degrees[rj] -= 1
        env.m_current -= 1
        env._edges_added_this_step = 0
        env.rr_update_remove(ri, rj)

        # Row-update for remove involved nodes O(N)
        involved_rem = list(set([ri, rj]))
        update_logits_for_nodes(
            policy, env, add_logits, involved_rem, n, 'add')
        update_logits_for_nodes(
            policy, env, rem_logits, involved_rem, n, 'remove')

        # === PER-SWAP EXACT REWARD ===
        assert env.m_current == env.m, \
            f"M violation: {env.m_current} != {env.m}"
        lambda2_after = exact_lambda2_np(env.adj)
        reward = lambda2_after - lambda2_cache
        lambda2_cache = lambda2_after

        swap_values.append(value.item())
        swap_log_probs.append(add_log_prob + rem_log_prob)
        swap_rewards.append(reward)
        swap_entropies.append(add_ent + rem_ent)
        swap_ctxs.append({
            'add_feat': add_feat.copy(),
            'add_mask': add_mask.copy(),
            'add_flat_idx': add_flat_idx,
            'rem_feat': rem_feat.copy(),
            'rem_mask': rem_mask.copy(),
            'rem_flat_idx': rem_flat_idx,
            'graph_feat': gf.copy(),
            'n': n,
        })

    final_lambda2 = algebraic_connectivity(env.adj)
    improvement = final_lambda2 - env.initial_lambda2

    metrics = {
        'lambda2': final_lambda2,
        'improvement': improvement,
        'total_reward': sum(swap_rewards),
        'mean_swap_reward': float(np.mean(swap_rewards)) if swap_rewards else 0,
        'entropy': float(np.mean(swap_entropies)) if swap_entropies else 0,
        'n': n,
        'm': m,
        'num_swaps_total': len(swap_rewards),
    }

    return (swap_values, swap_log_probs, swap_rewards,
            swap_entropies, swap_ctxs, metrics)


# ============================================================================
# PPO Training
# ============================================================================

def train_refine_pp(
    min_n: int = 8,
    max_n: int = 10,
    num_episodes: int = 20000,
    batch_size: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    entropy_coef_start: float = 0.1,
    entropy_coef_end: float = 0.01,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
    save_dir: str = None,
    checkpoint: str = None,
):
    """Train REFINE v9 (PP) with flat 3M swap loop and per-swap rewards."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = Path("logs") / f"refine_pp_{timestamp}"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_config = RefineConfig(hidden_dim=64, delta=1)

    # k_steps will be overridden per episode to 3*m
    env_config = RefineEnvConfig(
        min_n=min_n, max_n=max_n, k_steps=1, delta=1)
    env = RefineEnv(env_config, seed=seed)

    policy = RefinePolicy(policy_config).to(device)
    start_episode = 0

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['policy_state_dict'])
        start_episode = ckpt.get('episode', 0)
        print(f"Resumed from {checkpoint} (episode {start_episode})")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    num_updates = num_episodes // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=lr * 0.1
    )

    start_update = start_episode // batch_size
    for _ in range(start_update):
        scheduler.step()

    param_count = sum(p.numel() for p in policy.parameters())

    print("=" * 70)
    print("REFINE v9 (PP) — Flat 3M Swap Loop + Per-Swap Rewards")
    print("=" * 70)
    print(f"Graph size: n in [{min_n}, {max_n}]")
    print(f"Swap budget: 3*m per episode (flat loop)")
    print(f"Refresh: Lanczos every max(1, m//10) swaps")
    print(f"Reward: per-swap exact Δλ₂ (eigensolve per swap)")
    print(f"Episodes: {num_episodes} | Batch size: {batch_size} | "
          f"Updates: {num_updates}")
    print(f"Policy params: {param_count:,}")
    print(f"PPO: gamma={gamma}, clip={clip_eps}, gae_lambda={gae_lambda}, "
          f"epochs={ppo_epochs}")
    print(f"Entropy: {entropy_coef_start} -> {entropy_coef_end}")
    print(f"Device: {device}")
    print("=" * 70)

    metrics_history = []
    best_avg_improvement = -float('inf')
    t_start = time.time()
    episodes_done = start_episode

    pbar = tqdm(range(num_updates), desc="Training")

    for update in pbar:
        progress = update / max(num_updates - 1, 1)
        entropy_coef = (entropy_coef_start
                        + (entropy_coef_end - entropy_coef_start) * progress)

        # Collect batch of episodes (per-swap transitions)
        batch_values = []
        batch_log_probs = []
        batch_rewards = []
        batch_ctxs = []
        batch_entropies = []
        episode_boundaries = []
        batch_metrics = []

        for b in range(batch_size):
            n = random.randint(min_n, max_n)
            max_m = n * (n - 1) // 2
            min_m = n - 1
            m = random.randint(min_m, max_m)

            episode_boundaries.append(len(batch_values))

            (ep_values, ep_log_probs, ep_rewards,
             ep_entropies, ep_ctxs, ep_metrics) = collect_episode_pp(
                env, policy, n, m, device)

            batch_values.extend(ep_values)
            batch_log_probs.extend(ep_log_probs)
            batch_rewards.extend(ep_rewards)
            batch_entropies.extend(ep_entropies)
            batch_ctxs.extend(ep_ctxs)

            ep_metrics['episode'] = episodes_done + b
            batch_metrics.append(ep_metrics)

        episodes_done += batch_size

        if len(batch_values) == 0:
            continue

        # GAE per episode (over per-swap transitions)
        all_advantages = []
        all_returns = []

        for b in range(batch_size):
            start_idx = episode_boundaries[b]
            end_idx = (episode_boundaries[b + 1]
                       if b + 1 < batch_size else len(batch_values))

            ep_vals = batch_values[start_idx:end_idx]
            ep_rews = batch_rewards[start_idx:end_idx]

            if len(ep_vals) == 0:
                continue

            advantages = []
            gae = 0.0
            for i in reversed(range(len(ep_vals))):
                next_val = ep_vals[i + 1] if i + 1 < len(ep_vals) else 0.0
                td_delta = ep_rews[i] + gamma * next_val - ep_vals[i]
                gae = td_delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)

            returns = [a + v for a, v in zip(advantages, ep_vals)]

            adv_arr = np.array(advantages)
            if len(adv_arr) > 1:
                adv_arr = ((adv_arr - adv_arr.mean())
                           / (adv_arr.std() + 1e-8))
            all_advantages.extend(adv_arr.tolist())
            all_returns.extend(returns)

        if len(all_advantages) == 0:
            continue

        adv_t = torch.tensor(
            all_advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(
            all_returns, dtype=torch.float32, device=device)
        old_lp_t = torch.tensor(
            batch_log_probs[:len(all_advantages)],
            dtype=torch.float32, device=device)

        # PPO update (per-swap transitions)
        for _ in range(ppo_epochs):
            new_lps = []
            new_ents = []
            new_vals = []

            for ctx in batch_ctxs[:len(all_advantages)]:
                swap_lp, swap_ent, val = evaluate_swap(
                    policy, ctx, device)
                new_lps.append(swap_lp)
                new_ents.append(swap_ent)
                new_vals.append(val)

            new_lp = torch.stack(new_lps)
            new_ent = torch.stack(new_ents)
            new_val = torch.stack(new_vals)

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(
                ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_val, ret_t)
            entropy_loss = -new_ent.mean()

            loss = (policy_loss + 0.5 * value_loss
                    + entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        metrics_history.extend(batch_metrics)

        avg_l2 = np.mean([m['lambda2'] for m in batch_metrics])
        avg_imp = np.mean([m['improvement'] for m in batch_metrics])
        avg_swaps = np.mean([m['num_swaps_total'] for m in batch_metrics])
        pbar.set_postfix({
            'l2': f"{avg_l2:.2f}",
            'dl2': f"{avg_imp:.3f}",
            'swaps': f"{avg_swaps:.0f}",
            'ep': episodes_done,
        })

        if len(metrics_history) >= batch_size:
            recent = metrics_history[-batch_size * 10:]
            avg_imp_recent = np.mean([m['improvement'] for m in recent])
            if avg_imp_recent > best_avg_improvement:
                best_avg_improvement = avg_imp_recent
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'config': {
                        'hidden_dim': policy_config.hidden_dim,
                        'edge_feat_dim': policy_config.edge_feat_dim,
                        'graph_feat_dim': policy_config.graph_feat_dim,
                    },
                    'episode': episodes_done,
                    'metrics': {
                        'avg_improvement': avg_imp_recent,
                        'best_avg_improvement': best_avg_improvement,
                    },
                }, save_dir / "best.pt")

        log_freq = max(1, 100 // batch_size)
        if (update + 1) % log_freq == 0:
            recent = metrics_history[-batch_size * log_freq:]
            avg_l2_log = np.mean([m['lambda2'] for m in recent])
            avg_imp_log = np.mean([m['improvement'] for m in recent])
            avg_rew_log = np.mean([m['total_reward'] for m in recent])
            avg_swaps_log = np.mean(
                [m['num_swaps_total'] for m in recent])
            elapsed = time.time() - t_start
            eps_per_sec = (episodes_done - start_episode) / elapsed

            print(f"\n--- Update {update+1}/{num_updates} "
                  f"(ep {episodes_done}) [{elapsed/60:.1f}m] ---")
            print(f"  Avg l2: {avg_l2_log:.4f} | Avg dl2: {avg_imp_log:.4f}"
                  f" | Avg reward: {avg_rew_log:.4f}"
                  f" | Best: {best_avg_improvement:.4f}")
            print(f"  Avg swaps/ep: {avg_swaps_log:.0f} | "
                  f"{eps_per_sec:.1f} ep/s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Entropy coef: {entropy_coef:.4f}")

        ckpt_update_freq = max(1, 1000 // batch_size)
        if (update + 1) % ckpt_update_freq == 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'hidden_dim': policy_config.hidden_dim,
                    'edge_feat_dim': policy_config.edge_feat_dim,
                    'graph_feat_dim': policy_config.graph_feat_dim,
                },
                'episode': episodes_done,
            }, save_dir / f"checkpoint_{episodes_done}.pt")

    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'config': {
            'hidden_dim': policy_config.hidden_dim,
            'edge_feat_dim': policy_config.edge_feat_dim,
            'graph_feat_dim': policy_config.graph_feat_dim,
        },
        'episode': episodes_done,
    }, save_dir / "final.pt")

    with open(save_dir / "metrics.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print("REFINE v9 (PP) Training Complete!")
    print(f"  Episodes: {episodes_done} | Updates: {num_updates} "
          f"| Time: {total_time/60:.1f}m")
    print(f"  Best avg improvement: {best_avg_improvement:.4f}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*70}")

    return save_dir


def main():
    parser = argparse.ArgumentParser(
        description="REFINE v9 (PP): Flat 3M Swap Loop + Per-Swap Rewards")
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Episodes per PPO update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef-start", type=float, default=0.1)
    parser.add_argument("--entropy-coef-end", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    train_refine_pp(
        min_n=args.min_n,
        max_n=args.max_n,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ppo_epochs=args.ppo_epochs,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
