#!/usr/bin/env python3
"""
CRL Training — C-accelerated episode collection + Python PPO update.

Episode collection runs entirely in C (MLP inference, Lanczos, RR, sampling).
PPO gradient update stays in PyTorch (needs autograd).

Usage:
    python train.py --min-n 8 --max-n 16 --episodes 200000 --swap-frac 0.25
"""

import argparse
import ctypes
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from policy import RefinePolicy, RefineConfig

# ============================================================================
# C library loading
# ============================================================================

N_MAX = 24
EDGE_FEAT_DIM = 4
GRAPH_FEAT_DIM = 3
HIDDEN_DIM = 64
RR_K = 8


class MLP_C(ctypes.Structure):
    _fields_ = [
        ("W0", ctypes.c_float * (HIDDEN_DIM * EDGE_FEAT_DIM)),
        ("b0", ctypes.c_float * HIDDEN_DIM),
        ("W1", ctypes.c_float * (HIDDEN_DIM * HIDDEN_DIM)),
        ("b1", ctypes.c_float * HIDDEN_DIM),
        ("W2", ctypes.c_float * HIDDEN_DIM),
        ("b2", ctypes.c_float * 1),
        ("in_dim", ctypes.c_int),
    ]


class PolicyWeights(ctypes.Structure):
    _fields_ = [
        ("add_mlp", MLP_C),
        ("rem_mlp", MLP_C),
        ("val_mlp", MLP_C),
    ]


class SwapTxn(ctypes.Structure):
    _fields_ = [
        ("feat", ctypes.c_float * (N_MAX * N_MAX * EDGE_FEAT_DIM)),
        ("mask", ctypes.c_uint8 * (N_MAX * N_MAX)),
        ("flat_idx", ctypes.c_int32),
        ("gfeat", ctypes.c_float * GRAPH_FEAT_DIM),
        ("log_prob", ctypes.c_double),
        ("value", ctypes.c_double),
        ("reward", ctypes.c_double),
    ]


def load_crl_lib():
    d = Path(__file__).parent
    if sys.platform == "darwin":
        lib_path = d / "libcrl.dylib"
    else:
        lib_path = d / "libcrl.so"
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run 'make' first.")
        sys.exit(1)

    lib = ctypes.CDLL(str(lib_path))

    lib.crl_load_weights.restype = None
    lib.crl_load_weights.argtypes = [
        ctypes.POINTER(PolicyWeights),
    ] + [ctypes.POINTER(ctypes.c_float)] * 18

    lib.crl_collect_episode.restype = ctypes.c_int
    lib.crl_collect_episode.argtypes = [
        ctypes.POINTER(PolicyWeights),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint64,
        ctypes.POINTER(SwapTxn), ctypes.POINTER(SwapTxn),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]

    return lib


# ============================================================================
# Weight sync: PyTorch -> C
# ============================================================================

def sync_weights(lib, policy, pw):
    """Copy policy weights to C PolicyWeights struct."""
    sd = policy.state_dict()

    def get_arrays(prefix):
        W0 = sd[f'{prefix}.0.weight'].cpu().numpy().astype(np.float32).flatten()
        b0 = sd[f'{prefix}.0.bias'].cpu().numpy().astype(np.float32).flatten()
        W1 = sd[f'{prefix}.2.weight'].cpu().numpy().astype(np.float32).flatten()
        b1 = sd[f'{prefix}.2.bias'].cpu().numpy().astype(np.float32).flatten()
        W2 = sd[f'{prefix}.4.weight'].cpu().numpy().astype(np.float32).flatten()
        b2 = sd[f'{prefix}.4.bias'].cpu().numpy().astype(np.float32).flatten()
        return W0, b0, W1, b1, W2, b2

    add_arrays = get_arrays('add_mlp')
    rem_arrays = get_arrays('rem_mlp')
    val_arrays = get_arrays('value_mlp')

    ptrs = []
    # Keep references alive
    _refs = []
    for arrays in [add_arrays, rem_arrays, val_arrays]:
        for arr in arrays:
            _refs.append(arr)
            ptrs.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    lib.crl_load_weights(ctypes.byref(pw), *ptrs)
    return _refs  # prevent GC


# ============================================================================
# Transition extraction: C SwapTxn -> Python dict
# ============================================================================

def extract_txn(txn, n, phase='add'):
    """Convert a C SwapTxn to a Python dict for PPO replay."""
    feat = np.ctypeslib.as_array(txn.feat).reshape(N_MAX, N_MAX, EDGE_FEAT_DIM)
    mask = np.ctypeslib.as_array(txn.mask).reshape(N_MAX, N_MAX)
    gfeat = np.ctypeslib.as_array(txn.gfeat).copy()

    return {
        'feat': feat[:n, :n, :].copy(),
        'mask': mask[:n, :n].astype(bool),
        'flat_idx': int(txn.flat_idx),  # N_MAX-based
        'graph_feat': gfeat,
        'log_prob': float(txn.log_prob),
        'value': float(txn.value),
        'reward': float(txn.reward),
        'n': n,
        'phase': phase,
    }


# ============================================================================
# PPO re-evaluation
# ============================================================================

def evaluate_swap(policy, add_ctx, rem_ctx, device):
    """Re-evaluate one swap (add + rem) under current policy. Differentiable."""
    n = add_ctx['n']

    # Value from add_ctx graph features
    gf_t = torch.tensor(add_ctx['graph_feat'], dtype=torch.float32, device=device)
    value = policy.compute_value(gf_t)

    lps = []
    total_entropy = torch.tensor(0.0, device=device)

    # Re-evaluate ADD
    ef_add = torch.tensor(add_ctx['feat'], dtype=torch.float32, device=device)
    add_logits = policy.score_add(ef_add).view(-1)
    add_mask_t = torch.tensor(add_ctx['mask'], dtype=torch.bool, device=device).view(-1)
    valid_add = torch.where(add_mask_t)[0]

    if len(valid_add) > 0:
        valid_logits = add_logits[valid_add]
        lp = F.log_softmax(valid_logits, dim=0)
        p = torch.exp(lp)
        total_entropy = total_entropy - (p * lp).sum()

        # Find position of chosen index
        # flat_idx is N_MAX-based, convert to n-based
        flat_idx_nmax = add_ctx['flat_idx']
        ai = flat_idx_nmax // N_MAX
        aj = flat_idx_nmax % N_MAX
        flat_idx_n = ai * n + aj

        idx_to_pos = torch.full((n * n,), -1, dtype=torch.long, device=device)
        idx_to_pos[valid_add] = torch.arange(len(valid_add), device=device)
        pos = idx_to_pos[flat_idx_n]
        if pos >= 0:
            lps.append(lp[pos])

    # Re-evaluate REM
    ef_rem = torch.tensor(rem_ctx['feat'], dtype=torch.float32, device=device)
    rem_logits = policy.score_remove(ef_rem).view(-1)
    rem_mask_t = torch.tensor(rem_ctx['mask'], dtype=torch.bool, device=device).view(-1)
    valid_rem = torch.where(rem_mask_t)[0]

    if len(valid_rem) > 0:
        valid_logits = rem_logits[valid_rem]
        lp = F.log_softmax(valid_logits, dim=0)
        p = torch.exp(lp)
        total_entropy = total_entropy - (p * lp).sum()

        flat_idx_nmax = rem_ctx['flat_idx']
        ri = flat_idx_nmax // N_MAX
        rj = flat_idx_nmax % N_MAX
        flat_idx_n = ri * n + rj

        idx_to_pos = torch.full((n * n,), -1, dtype=torch.long, device=device)
        idx_to_pos[valid_rem] = torch.arange(len(valid_rem), device=device)
        pos = idx_to_pos[flat_idx_n]
        if pos >= 0:
            lps.append(lp[pos])

    if lps:
        total_lp = torch.stack(lps).sum()
    else:
        total_lp = torch.tensor(0.0, device=device, requires_grad=True)

    return total_lp, total_entropy, value


# ============================================================================
# Training loop
# ============================================================================

def train_crl(
    min_n=8, max_n=16, num_episodes=200000, k_steps=20,
    batch_size=8, swap_frac=0.25,
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
    ppo_epochs=4, entropy_coef_start=0.1, entropy_coef_end=0.01,
    grad_clip=1.0, device="cpu", seed=42, save_dir=None, checkpoint=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    lib = load_crl_lib()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = Path("logs") / f"crl_{timestamp}"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = RefineConfig(hidden_dim=64, delta=1)
    policy = RefinePolicy(config).to(device)
    start_episode = 0

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['policy_state_dict'])
        start_episode = ckpt.get('episode', 0)
        print(f"Resumed from {checkpoint} (episode {start_episode})")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    num_updates = num_episodes // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=lr * 0.1)

    start_update = start_episode // batch_size
    for _ in range(start_update):
        scheduler.step()

    pw = PolicyWeights()
    max_txns = k_steps * max(1, int(max_n * (max_n - 1) // 2 * swap_frac)) + 100

    param_count = sum(p.numel() for p in policy.parameters())

    print("=" * 70)
    print("CRL — C-Accelerated RL Training")
    print("=" * 70)
    print(f"Graph size: n in [{min_n}, {max_n}]")
    print(f"K steps: {k_steps} | swap_frac: {swap_frac}")
    print(f"Episodes: {num_episodes} | Batch: {batch_size} | Updates: {num_updates}")
    print(f"Policy params: {param_count:,}")
    print(f"Max txns/episode: {max_txns}")
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

        # Sync weights to C
        _refs = sync_weights(lib, policy, pw)

        # Collect batch of episodes via C
        batch_add_ctxs = []
        batch_rem_ctxs = []
        batch_values = []
        batch_log_probs = []
        batch_rewards = []
        episode_boundaries = []
        batch_metrics = []

        for b in range(batch_size):
            n = random.randint(min_n, max_n)
            max_m = n * (n - 1) // 2
            min_m = n - 1
            m_val = random.randint(min_m, max_m)

            episode_boundaries.append(len(batch_values))

            ep_seed = seed + episodes_done + b

            add_txns = (SwapTxn * max_txns)()
            rem_txns = (SwapTxn * max_txns)()
            out_na = ctypes.c_int(0)
            out_nr = ctypes.c_int(0)
            out_metrics = (ctypes.c_double * 5)()

            ret = lib.crl_collect_episode(
                ctypes.byref(pw),
                ctypes.c_int(n), ctypes.c_int(m_val),
                ctypes.c_int(k_steps), ctypes.c_double(swap_frac),
                ctypes.c_uint64(ep_seed),
                add_txns, rem_txns,
                ctypes.byref(out_na), ctypes.byref(out_nr),
                out_metrics, ctypes.c_int(max_txns)
            )

            if ret != 0:
                continue

            na = out_na.value
            nr = out_nr.value
            num_swaps = min(na, nr)

            for i in range(num_swaps):
                a_ctx = extract_txn(add_txns[i], n, 'add')
                r_ctx = extract_txn(rem_txns[i], n, 'rem')

                batch_add_ctxs.append(a_ctx)
                batch_rem_ctxs.append(r_ctx)
                batch_values.append(a_ctx['value'])
                batch_log_probs.append(a_ctx['log_prob'] + r_ctx['log_prob'])
                batch_rewards.append(a_ctx['reward'])

            ep_metrics = {
                'lambda2': out_metrics[0],
                'improvement': out_metrics[2],
                'total_reward': sum(add_txns[i].reward for i in range(na)),
                'num_swaps_total': num_swaps,
                'n': n, 'm': m_val,
                'episode': episodes_done + b,
            }
            batch_metrics.append(ep_metrics)

        episodes_done += batch_size
        del _refs  # allow GC of weight arrays

        if len(batch_values) == 0:
            continue

        # GAE per episode
        all_advantages = []
        all_returns = []

        for b_idx in range(batch_size):
            start_idx = episode_boundaries[b_idx]
            end_idx = (episode_boundaries[b_idx + 1]
                       if b_idx + 1 < batch_size else len(batch_values))

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
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
            all_advantages.extend(adv_arr.tolist())
            all_returns.extend(returns)

        if len(all_advantages) == 0:
            continue

        adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
        old_lp_t = torch.tensor(
            batch_log_probs[:len(all_advantages)],
            dtype=torch.float32, device=device)

        # PPO update
        for _ in range(ppo_epochs):
            new_lps = []
            new_ents = []
            new_vals = []

            for idx in range(len(all_advantages)):
                swap_lp, swap_ent, val = evaluate_swap(
                    policy, batch_add_ctxs[idx], batch_rem_ctxs[idx], device)
                new_lps.append(swap_lp)
                new_ents.append(swap_ent)
                new_vals.append(val)

            new_lp = torch.stack(new_lps)
            new_ent = torch.stack(new_ents)
            new_val = torch.stack(new_vals)

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_val, ret_t)
            entropy_loss = -new_ent.mean()

            loss = policy_loss + 0.5 * value_loss + entropy_coef * entropy_loss

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

        # Best checkpoint
        if len(metrics_history) >= batch_size:
            recent = metrics_history[-batch_size * 10:]
            avg_imp_recent = np.mean([m['improvement'] for m in recent])
            if avg_imp_recent > best_avg_improvement:
                best_avg_improvement = avg_imp_recent
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'config': {
                        'hidden_dim': config.hidden_dim,
                        'edge_feat_dim': config.edge_feat_dim,
                        'graph_feat_dim': config.graph_feat_dim,
                    },
                    'episode': episodes_done,
                    'metrics': {
                        'avg_improvement': avg_imp_recent,
                        'best_avg_improvement': best_avg_improvement,
                    },
                }, save_dir / "best.pt")

        # Periodic logging
        log_freq = max(1, 100 // batch_size)
        if (update + 1) % log_freq == 0:
            recent = metrics_history[-batch_size * log_freq:]
            avg_l2_log = np.mean([m['lambda2'] for m in recent])
            avg_imp_log = np.mean([m['improvement'] for m in recent])
            elapsed = time.time() - t_start
            eps_per_sec = (episodes_done - start_episode) / elapsed

            print(f"\n--- Update {update+1}/{num_updates} "
                  f"(ep {episodes_done}) [{elapsed/60:.1f}m] ---")
            print(f"  Avg l2: {avg_l2_log:.4f} | Avg dl2: {avg_imp_log:.4f}"
                  f" | Best: {best_avg_improvement:.4f}")
            print(f"  Avg swaps/ep: {avg_swaps:.0f} | "
                  f"{eps_per_sec:.1f} ep/s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Periodic checkpoint
        ckpt_freq = max(1, 1000 // batch_size)
        if (update + 1) % ckpt_freq == 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'hidden_dim': config.hidden_dim,
                    'edge_feat_dim': config.edge_feat_dim,
                    'graph_feat_dim': config.graph_feat_dim,
                },
                'episode': episodes_done,
            }, save_dir / f"checkpoint_{episodes_done}.pt")

    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'config': {
            'hidden_dim': config.hidden_dim,
            'edge_feat_dim': config.edge_feat_dim,
            'graph_feat_dim': config.graph_feat_dim,
        },
        'episode': episodes_done,
    }, save_dir / "final.pt")

    with open(save_dir / "metrics.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print("CRL Training Complete!")
    print(f"  Episodes: {episodes_done} | Updates: {num_updates} "
          f"| Time: {total_time/60:.1f}m")
    print(f"  Best avg improvement: {best_avg_improvement:.4f}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*70}")

    return save_dir


def main():
    parser = argparse.ArgumentParser(description="CRL: C-Accelerated RL Training")
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--k-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--swap-frac", type=float, default=0.25)
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
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    train_crl(
        min_n=args.min_n, max_n=args.max_n,
        num_episodes=args.episodes, k_steps=args.k_steps,
        batch_size=args.batch_size, swap_frac=args.swap_frac,
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        grad_clip=args.grad_clip, device=args.device,
        seed=args.seed, save_dir=args.save_dir, checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
