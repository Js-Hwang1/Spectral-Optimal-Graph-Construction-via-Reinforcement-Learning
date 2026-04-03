#!/usr/bin/env python3
"""
REFINE v9 — Aligned PPO: Per-Edge Features + Batch Rewards.

Fixes the PPO mismatch in v7-rescore: each edge op rebuilds full features
and stores them, so PPO re-evaluation uses the exact state the agent saw
when it chose each action. Rewards remain batch (one eigensolve per phase).

This combines:
  - v8's PPO alignment (per-edge features → correct policy gradients)
  - v7's batch rewards (no per-edge myopia → better edge combinations)

Each K-step:
  Phase 1 (ADD):
    For each edge: rebuild features → MLP → sample → execute → RR update
    Store per-edge: (features, mask, action, log_prob, value)
    One eigensolve after all adds → r_add assigned to LAST add transition
    GAE propagates credit backward through the batch

  Phase 2 (REM):
    Fresh RR on post-add graph
    Same per-edge pattern
    r_rem = λ₂(post-rem) - λ₂(post-add) on LAST rem transition

Usage:
    python train_refine_batch.py --min-n 8 --max-n 16 --episodes 50000
"""

import argparse
import json
import multiprocessing as mp
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

# Globals for fork-based parallel workers (COW shared from parent)
_g_policy = None
_g_env_cfg_dict = None
_g_k_steps = 20
_g_swap_frac = 0.05


# ============================================================================
# Numpy MLP inference — same computation, no torch dispatch overhead
# ============================================================================

def _extract_numpy_weights(mlp):
    """Extract MLP weights as numpy arrays for fast collection inference."""
    layers = []
    for module in mlp:
        if isinstance(module, torch.nn.Linear):
            layers.append((
                module.weight.detach().numpy(),   # (out, in)
                module.bias.detach().numpy(),      # (out,)
            ))
    return layers


def _numpy_mlp_forward(layers, x):
    """Forward pass in pure numpy. x: (..., feat_dim) -> (..., 1)."""
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:  # SiLU after all but last layer
            x = x * (1.0 / (1.0 + np.exp(np.clip(-x, -50, 50))))
    return x.reshape(orig_shape + (-1,) if x.shape[-1] > 1 else orig_shape)


# ============================================================================
# Build (N,N,10) features from current RR state — O(N²)
# ============================================================================

def build_features(env, n):
    """Build (N,N,4) features from current env RR state.

    Only Fiedler gap (r=0.90 with Δλ₂), degrees, and step fraction.
    v3/v4/v5 gaps and spectral gap scalars removed — PoC showed they add noise.
    """
    v2, _, lam2, _ = env.get_rr_spectral_info()

    nm1 = max(n - 1, 1)
    deg_norm = env.degrees / nm1
    step_frac = env.step / max(env.config.k_steps, 1)

    feat = np.zeros((n, n, 4), dtype=np.float32)
    feat[:, :, 0] = n * (v2[:, None] - v2[None, :]) ** 2
    feat[:, :, 1] = deg_norm[:, None]
    feat[:, :, 2] = deg_norm[None, :]
    feat[:, :, 3] = step_frac
    return feat


def get_graph_feat(env):
    """Graph-level features (3-dim) from current RR state."""
    _, _, lam2, _ = env.get_rr_spectral_info()
    n = env.n
    nm1 = max(n - 1, 1)
    deg_mean = (env.degrees / nm1).mean()
    step_frac = env.step / max(env.config.k_steps, 1)
    return np.array([
        deg_mean, lam2 / max(n, 1), step_frac
    ], dtype=np.float32)


# ============================================================================
# Batched PPO re-evaluation — one MLP call per episode
# ============================================================================

def evaluate_add_batch(policy, txns, device):
    """Re-evaluate ADD transitions from one episode (same n). Batched MLP."""
    T = len(txns)

    feats = np.stack([t['feat'] for t in txns])  # (T, N, N, 10)
    feats_t = torch.tensor(feats, dtype=torch.float32, device=device)
    logits = policy.score_add(feats_t)  # (T, N, N)

    gfs = np.stack([t['graph_feat'] for t in txns])  # (T, 9)
    gfs_t = torch.tensor(gfs, dtype=torch.float32, device=device)
    values = policy.compute_value(gfs_t)  # (T,)

    log_probs = []
    entropies = []
    for t in range(T):
        mask_flat = torch.tensor(
            txns[t]['mask'].reshape(-1), dtype=torch.bool, device=device)
        valid = torch.where(mask_flat)[0]

        if len(valid) == 0:
            log_probs.append(
                torch.tensor(0.0, device=device, requires_grad=True))
            entropies.append(torch.tensor(0.0, device=device))
            continue

        vl = torch.clamp(logits[t].view(-1)[valid], -50, 50)
        lp = F.log_softmax(vl, dim=0)
        p = torch.exp(lp)
        entropies.append(-(p * lp).sum())

        flat_idx = txns[t]['flat_idx']
        pos = (valid == flat_idx).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            log_probs.append(lp[pos[0]])
        else:
            log_probs.append(
                torch.tensor(0.0, device=device, requires_grad=True))

    return torch.stack(log_probs), torch.stack(entropies), values


def evaluate_rem_batch(policy, txns, device):
    """Re-evaluate REM transitions from one episode (same n). Batched MLP."""
    T = len(txns)

    feats = np.stack([t['feat'] for t in txns])
    feats_t = torch.tensor(feats, dtype=torch.float32, device=device)
    logits = policy.score_remove(feats_t)

    gfs = np.stack([t['graph_feat'] for t in txns])
    gfs_t = torch.tensor(gfs, dtype=torch.float32, device=device)
    values = policy.compute_value(gfs_t)

    log_probs = []
    entropies = []
    for t in range(T):
        mask_flat = torch.tensor(
            txns[t]['mask'].reshape(-1), dtype=torch.bool, device=device)
        valid = torch.where(mask_flat)[0]

        if len(valid) == 0:
            log_probs.append(
                torch.tensor(0.0, device=device, requires_grad=True))
            entropies.append(torch.tensor(0.0, device=device))
            continue

        vl = torch.clamp(logits[t].view(-1)[valid], -50, 50)
        lp = F.log_softmax(vl, dim=0)
        p = torch.exp(lp)
        entropies.append(-(p * lp).sum())

        flat_idx = txns[t]['flat_idx']
        pos = (valid == flat_idx).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            log_probs.append(lp[pos[0]])
        else:
            log_probs.append(
                torch.tensor(0.0, device=device, requires_grad=True))

    return torch.stack(log_probs), torch.stack(entropies), values


# ============================================================================
# Episode collection — per-edge features, batch rewards
# ============================================================================

def collect_episode(env, policy, n, m, k_steps, device, swap_frac=0.05,
                    np_add=None, np_rem=None, np_val=None):
    """
    Collect one episode with per-edge feature snapshots + batch rewards.

    If np_add/np_rem/np_val are provided (numpy MLP weights), uses pure numpy
    for inference (same computation, ~2× faster by avoiding torch dispatch).
    """
    env.reset(n=n, m=m)

    add_txns = []
    rem_txns = []
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    num_swaps = max(1, int(m * swap_frac))
    total_add_reward = 0.0
    total_rem_reward = 0.0

    for k in range(k_steps):
        env.step = k
        env.init_rr()
        lambda2_pre = exact_lambda2_np(env.adj)

        # ==========================================================
        # PHASE 1: ADD — per-edge features, batch reward
        # ==========================================================
        env.begin_add_phase()
        k_add_txns = []

        for s in range(num_swaps):
            feat = build_features(env, n)
            gf = get_graph_feat(env)

            cur_mask = (env.adj == 0) & upper_tri
            valid = np.where(cur_mask.reshape(-1))[0]
            if len(valid) == 0:
                break

            ef_t = torch.tensor(feat, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = policy.score_add(ef_t).cpu().numpy()

            cur_logits_t = torch.tensor(
                logits.reshape(-1)[valid], dtype=torch.float32)
            cur_logits_t = torch.clamp(cur_logits_t, -50, 50)
            cur_probs = F.softmax(cur_logits_t, dim=0)
            if torch.isnan(cur_probs).any() or (cur_probs < 0).any():
                cur_probs = torch.ones_like(cur_probs) / len(cur_probs)

            sel = torch.multinomial(cur_probs, 1)
            flat_idx = int(valid[sel.item()])
            log_prob = torch.log(cur_probs[sel]).item()

            with torch.no_grad():
                gf_t = torch.tensor(gf, dtype=torch.float32, device=device)
                value = policy.compute_value(gf_t).item()

            k_add_txns.append({
                'feat': feat,
                'mask': cur_mask,
                'flat_idx': flat_idx,
                'graph_feat': gf,
                'log_prob': log_prob,
                'value': value,
                'reward': 0.0,
                'n': n,
            })

            ai, aj = flat_idx // n, flat_idx % n
            env.add_single_edge(ai, aj)

        if not k_add_txns:
            continue

        num_added = len(k_add_txns)
        add_flat_indices = [t['flat_idx'] for t in k_add_txns]

        # Check removability, trim excess adds if needed
        bridge_mask = env.get_bridge_mask()
        rem_check = (env.adj > 0) & upper_tri & ~bridge_mask
        for fi in add_flat_indices:
            ai, aj = fi // n, fi % n
            ai2, aj2 = min(ai, aj), max(ai, aj)
            rem_check[ai2, aj2] = False
        available = int(rem_check.sum())

        if available == 0:
            for t in reversed(k_add_txns):
                ai, aj = t['flat_idx'] // n, t['flat_idx'] % n
                env.adj[ai, aj] = 0; env.adj[aj, ai] = 0
                env.degrees[ai] -= 1; env.degrees[aj] -= 1
                env.m_current -= 1
                env.rr_update_remove(ai, aj)
            continue

        effective = min(num_added, available)
        if effective < num_added:
            excess = num_added - effective
            for t in reversed(k_add_txns[-excess:]):
                ai, aj = t['flat_idx'] // n, t['flat_idx'] % n
                env.adj[ai, aj] = 0; env.adj[aj, ai] = 0
                env.degrees[ai] -= 1; env.degrees[aj] -= 1
                env.m_current -= 1
                env.rr_update_remove(ai, aj)
            k_add_txns = k_add_txns[:effective]
            add_flat_indices = add_flat_indices[:effective]

        # Batch eigensolve after effective adds
        lambda2_post_add = exact_lambda2_np(env.adj)
        r_add = lambda2_post_add - lambda2_pre
        k_add_txns[-1]['reward'] = r_add
        total_add_reward += r_add
        add_txns.extend(k_add_txns)

        # ==========================================================
        # PHASE 2: REM — per-edge features, batch reward (isolated)
        # ==========================================================
        env.init_rr()  # Fresh RR on post-add graph

        bridge_mask = env.get_bridge_mask()
        base_rem_mask = (env.adj > 0) & upper_tri & ~bridge_mask
        for fi in add_flat_indices:
            ai, aj = fi // n, fi % n
            ai2, aj2 = min(ai, aj), max(ai, aj)
            base_rem_mask[ai2, aj2] = False

        k_rem_txns = []
        rem_mask_live = base_rem_mask.copy()

        for s in range(effective):
            feat = build_features(env, n)
            gf = get_graph_feat(env)

            valid = np.where(rem_mask_live.reshape(-1))[0]
            if len(valid) == 0:
                break

            ef_t = torch.tensor(feat, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = policy.score_remove(ef_t).cpu().numpy()

            cur_logits_t = torch.tensor(
                logits.reshape(-1)[valid], dtype=torch.float32)
            cur_logits_t = torch.clamp(cur_logits_t, -50, 50)
            cur_probs = F.softmax(cur_logits_t, dim=0)
            if torch.isnan(cur_probs).any() or (cur_probs < 0).any():
                cur_probs = torch.ones_like(cur_probs) / len(cur_probs)

            sel = torch.multinomial(cur_probs, 1)
            flat_idx = int(valid[sel.item()])
            log_prob = torch.log(cur_probs[sel]).item()

            with torch.no_grad():
                gf_t = torch.tensor(gf, dtype=torch.float32, device=device)
                value = policy.compute_value(gf_t).item()

            k_rem_txns.append({
                'feat': feat,
                'mask': rem_mask_live.copy(),
                'flat_idx': flat_idx,
                'graph_feat': gf,
                'log_prob': log_prob,
                'value': value,
                'reward': 0.0,
                'n': n,
            })

            ri, rj = flat_idx // n, flat_idx % n
            env.adj[ri, rj] = 0; env.adj[rj, ri] = 0
            env.degrees[ri] -= 1; env.degrees[rj] -= 1
            env.m_current -= 1
            env._edges_added_this_step = 0
            env.rr_update_remove(ri, rj)
            rem_mask_live[ri, rj] = False

        actual_rems = len(k_rem_txns)

        # Undo unmatched adds (rare)
        if actual_rems < effective:
            unmatched = effective - actual_rems
            for t in reversed(k_add_txns[-unmatched:]):
                ai, aj = t['flat_idx'] // n, t['flat_idx'] % n
                if env.adj[ai, aj] > 0:
                    env.adj[ai, aj] = 0; env.adj[aj, ai] = 0
                    env.degrees[ai] -= 1; env.degrees[aj] -= 1
                    env.m_current -= 1
                    env.rr_update_remove(ai, aj)
            # Remove unmatched add txns
            for _ in range(unmatched):
                add_txns.pop()

        assert env.m_current == env.m, \
            f"M mismatch: {env.m_current} != {env.m}"

        if k_rem_txns:
            lambda2_post_rem = exact_lambda2_np(env.adj)
            r_rem = lambda2_post_rem - lambda2_post_add
            k_rem_txns[-1]['reward'] = r_rem
            total_rem_reward += r_rem
            rem_txns.extend(k_rem_txns)
            env.lambda2 = lambda2_post_rem
        else:
            env.lambda2 = lambda2_post_add

    final_lambda2 = algebraic_connectivity(env.adj)
    improvement = final_lambda2 - env.initial_lambda2

    metrics = {
        'lambda2': final_lambda2,
        'improvement': improvement,
        'add_reward_total': total_add_reward,
        'rem_reward_mean': total_rem_reward / max(k_steps, 1),
        'rem_reward_total': total_rem_reward,
        'n': n,
        'm': m,
        'num_add_txns': len(add_txns),
        'num_rem_txns': len(rem_txns),
    }

    return add_txns, rem_txns, metrics


# ============================================================================
# Parallel episode collection worker (fork-based, COW shared policy)
# ============================================================================

def _fork_collect_worker(args):
    """Fork worker: uses COW-shared _g_policy from parent. No pickling."""
    n, m, seed = args
    env = RefineEnv(RefineEnvConfig(**_g_env_cfg_dict), seed=seed)
    return collect_episode(env, _g_policy, n, m, _g_k_steps, 'cpu', _g_swap_frac)


# ============================================================================
# GAE computation
# ============================================================================

def compute_gae(values, rewards, gamma, gae_lambda):
    """Compute GAE advantages and returns."""
    advantages = []
    gae = 0.0
    for i in reversed(range(len(values))):
        next_val = values[i + 1] if i + 1 < len(values) else 0.0
        td_delta = rewards[i] + gamma * next_val - values[i]
        gae = td_delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    adv_arr = np.array(advantages)
    if len(adv_arr) > 1:
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
    return adv_arr.tolist(), returns


# ============================================================================
# PPO Training
# ============================================================================

def train_refine_batch(
    min_n: int = 8,
    max_n: int = 16,
    num_episodes: int = 50000,
    k_steps: int = 20,
    batch_size: int = 8,
    swap_frac: float = 0.05,
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
    init_method: str = "ring",
    num_workers: int = 1,
):
    """Train with aligned PPO: per-edge features + batch rewards."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = Path("logs") / f"refine_v9_{timestamp}"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_config = RefineConfig(hidden_dim=64, delta=1)
    env_config = RefineEnvConfig(
        min_n=min_n, max_n=max_n, k_steps=k_steps, delta=1,
        init_method=init_method)
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
    print("REFINE v9 — Aligned PPO: Per-Edge Features + Batch Rewards")
    print("=" * 70)
    print(f"Graph size: n in [{min_n}, {max_n}]")
    print(f"Init method: {init_method}")
    print(f"K steps: {k_steps} | swaps/step: max(1, int(m*{swap_frac}))")
    print(f"Per-edge features: full rebuild before each action (PPO-aligned)")
    print(f"ADD reward: Δλ₂_add = λ₂(post-add) - λ₂(pre)  [on last edge]")
    print(f"REM reward: Δλ₂_rem = λ₂(post-rem) - λ₂(post-add)  [on last edge]")
    print(f"Eigensolves: 2 per K-step (batch, like v7)")
    print(f"Episodes: {num_episodes} | Batch size: {batch_size} | "
          f"Updates: {num_updates}")
    print(f"Policy params: {param_count:,} (ADD+REM MLPs independent)")
    print(f"PPO: gamma={gamma}, clip={clip_eps}, gae_lambda={gae_lambda}, "
          f"epochs={ppo_epochs}")
    print(f"Entropy: {entropy_coef_start} -> {entropy_coef_end}")
    print(f"Device: {device}")
    print(f"Workers: {num_workers}")
    print("=" * 70)

    # Setup fork-based parallel collection with shared-memory policy
    use_parallel = num_workers > 1
    pool = None
    if use_parallel:
        global _g_policy, _g_env_cfg_dict, _g_k_steps, _g_swap_frac
        _g_env_cfg_dict = {
            'min_n': min_n, 'max_n': max_n,
            'k_steps': k_steps, 'delta': 1,
            'init_method': init_method,
        }
        _g_k_steps = k_steps
        _g_swap_frac = swap_frac
        policy.share_memory()  # Parameters in shared mem → visible to forks
        _g_policy = policy
        pool = mp.get_context('fork').Pool(num_workers)

    metrics_history = []
    best_avg_improvement = -float('inf')
    t_start = time.time()
    episodes_done = start_episode

    pbar = tqdm(range(num_updates), desc="Training")

    for update in pbar:
        progress = update / max(num_updates - 1, 1)
        entropy_coef = (entropy_coef_start
                        + (entropy_coef_end - entropy_coef_start) * progress)

        # --- Collect batch of episodes ---
        episodes_add = []
        episodes_rem = []
        batch_metrics = []

        ep_configs = []
        for b in range(batch_size):
            n = random.randint(min_n, max_n)
            max_m = n * (n - 1) // 2
            min_m = n - 1
            m = random.randint(min_m, max_m)
            ep_configs.append((n, m, episodes_done + b))

        if use_parallel:
            results = pool.map(_fork_collect_worker, ep_configs)
            for b, (add_txns, rem_txns, ep_metrics) in enumerate(results):
                episodes_add.append(add_txns)
                episodes_rem.append(rem_txns)
                ep_metrics['episode'] = episodes_done + b
                batch_metrics.append(ep_metrics)
        else:
            for b, (n, m, ep_seed) in enumerate(ep_configs):
                add_txns, rem_txns, ep_metrics = collect_episode(
                    env, policy, n, m, k_steps, device, swap_frac=swap_frac)
                episodes_add.append(add_txns)
                episodes_rem.append(rem_txns)
                ep_metrics['episode'] = episodes_done + b
                batch_metrics.append(ep_metrics)

        episodes_done += batch_size

        # --- GAE per episode, separate ADD/REM streams ---
        for ep_txns in episodes_add:
            if not ep_txns:
                continue
            vals = [t['value'] for t in ep_txns]
            rews = [t['reward'] for t in ep_txns]
            advs, rets = compute_gae(vals, rews, gamma, gae_lambda)
            for t, a, r in zip(ep_txns, advs, rets):
                t['advantage'] = a
                t['return'] = r

        for ep_txns in episodes_rem:
            if not ep_txns:
                continue
            vals = [t['value'] for t in ep_txns]
            rews = [t['reward'] for t in ep_txns]
            advs, rets = compute_gae(vals, rews, gamma, gae_lambda)
            for t, a, r in zip(ep_txns, advs, rets):
                t['advantage'] = a
                t['return'] = r

        # Check we have transitions
        has_add = any(len(ep) > 0 for ep in episodes_add)
        has_rem = any(len(ep) > 0 for ep in episodes_rem)
        if not has_add and not has_rem:
            continue

        # --- PPO update ---
        for _ in range(ppo_epochs):
            loss = torch.tensor(0.0, device=device)

            # --- ADD loss ---
            if has_add:
                all_lps, all_ents, all_vals = [], [], []
                all_old_lps, all_advs, all_rets = [], [], []

                for ep_txns in episodes_add:
                    if not ep_txns:
                        continue
                    new_lps, new_ents, new_vals = evaluate_add_batch(
                        policy, ep_txns, device)
                    all_lps.append(new_lps)
                    all_ents.append(new_ents)
                    all_vals.append(new_vals)
                    all_old_lps.append(torch.tensor(
                        [t['log_prob'] for t in ep_txns],
                        dtype=torch.float32, device=device))
                    all_advs.append(torch.tensor(
                        [t['advantage'] for t in ep_txns],
                        dtype=torch.float32, device=device))
                    all_rets.append(torch.tensor(
                        [t['return'] for t in ep_txns],
                        dtype=torch.float32, device=device))

                if all_lps:
                    lps = torch.cat(all_lps)
                    old_lps = torch.cat(all_old_lps)
                    advs = torch.cat(all_advs)
                    rets = torch.cat(all_rets)
                    ents = torch.cat(all_ents)
                    vals = torch.cat(all_vals)

                    ratio = torch.exp(lps - old_lps)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(
                        ratio, 1 - clip_eps, 1 + clip_eps) * advs
                    add_pol = -torch.min(surr1, surr2).mean()
                    add_val = F.mse_loss(vals, rets)
                    add_ent = -ents.mean()

                    loss = loss + add_pol + 0.5 * add_val \
                        + entropy_coef * add_ent

            # --- REM loss ---
            if has_rem:
                all_lps, all_ents, all_vals = [], [], []
                all_old_lps, all_advs, all_rets = [], [], []

                for ep_txns in episodes_rem:
                    if not ep_txns:
                        continue
                    new_lps, new_ents, new_vals = evaluate_rem_batch(
                        policy, ep_txns, device)
                    all_lps.append(new_lps)
                    all_ents.append(new_ents)
                    all_vals.append(new_vals)
                    all_old_lps.append(torch.tensor(
                        [t['log_prob'] for t in ep_txns],
                        dtype=torch.float32, device=device))
                    all_advs.append(torch.tensor(
                        [t['advantage'] for t in ep_txns],
                        dtype=torch.float32, device=device))
                    all_rets.append(torch.tensor(
                        [t['return'] for t in ep_txns],
                        dtype=torch.float32, device=device))

                if all_lps:
                    lps = torch.cat(all_lps)
                    old_lps = torch.cat(all_old_lps)
                    advs = torch.cat(all_advs)
                    rets = torch.cat(all_rets)
                    ents = torch.cat(all_ents)
                    vals = torch.cat(all_vals)

                    ratio = torch.exp(lps - old_lps)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(
                        ratio, 1 - clip_eps, 1 + clip_eps) * advs
                    rem_pol = -torch.min(surr1, surr2).mean()
                    rem_val = F.mse_loss(vals, rets)
                    rem_ent = -ents.mean()

                    loss = loss + rem_pol + 0.5 * rem_val \
                        + entropy_coef * rem_ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # --- Logging ---
        metrics_history.extend(batch_metrics)

        avg_l2 = np.mean([m['lambda2'] for m in batch_metrics])
        avg_imp = np.mean([m['improvement'] for m in batch_metrics])
        avg_rem = np.mean([m['rem_reward_mean'] for m in batch_metrics])
        pbar.set_postfix({
            'l2': f"{avg_l2:.2f}",
            'dl2': f"{avg_imp:.3f}",
            'rem': f"{avg_rem:.3f}",
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
            avg_add_r = np.mean([m['add_reward_total'] for m in recent])
            avg_rem_r = np.mean([m['rem_reward_mean'] for m in recent])
            avg_add_txns = np.mean(
                [m.get('num_add_txns', 0) for m in recent])
            elapsed = time.time() - t_start
            eps_per_sec = (episodes_done - start_episode) / elapsed

            print(f"\n--- Update {update+1}/{num_updates} "
                  f"(ep {episodes_done}) [{elapsed/60:.1f}m] ---")
            print(f"  Avg l2: {avg_l2_log:.4f} | Avg dl2: {avg_imp_log:.4f}"
                  f" | Best: {best_avg_improvement:.4f}")
            print(f"  ADD reward: {avg_add_r:.4f}"
                  f" | REM Δλ₂: {avg_rem_r:.4f}"
                  f" | Avg txns/ep: {avg_add_txns:.0f}")
            print(f"  {eps_per_sec:.1f} ep/s | "
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

    if pool is not None:
        pool.close()
        pool.join()

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
    print("REFINE v9 Training Complete!")
    print(f"  Episodes: {episodes_done} | Updates: {num_updates} "
          f"| Time: {total_time/60:.1f}m")
    print(f"  Best avg improvement: {best_avg_improvement:.4f}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*70}")

    return save_dir


def main():
    parser = argparse.ArgumentParser(
        description="REFINE v9: Aligned PPO + Batch Rewards")
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--k-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Episodes per PPO update")
    parser.add_argument("--swap-frac", type=float, default=0.05,
                        help="Fraction of edges to swap per step")
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
    parser.add_argument("--init", type=str, default="ring",
                        choices=["ring", "fv"],
                        help="Graph init: ring (ring+random) or fv (FV greedy)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel episode collection workers (1=sequential)")
    args = parser.parse_args()

    train_refine_batch(
        min_n=args.min_n,
        max_n=args.max_n,
        num_episodes=args.episodes,
        k_steps=args.k_steps,
        batch_size=args.batch_size,
        swap_frac=args.swap_frac,
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
        init_method=args.init,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
