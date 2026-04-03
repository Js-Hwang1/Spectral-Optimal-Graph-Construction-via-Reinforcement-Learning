#!/usr/bin/env python3
"""
Train G(n,m) Model with PPO + GNN Edge Selection.

GNN encoder propagates structural info via message passing on adjacency.
Features: degree + 2-hop + 3-hop (no Lanczos/eigensolvers at inference).
R = m//4 swaps per step — scales with edge count.

Per step (K steps/episode):
  1. Compute node features (3-dim) + adj tensor              [O(N²)]
  2. GNN encode + score edges/non-edges                      [O(N²)]
  3. Apply R swaps (no BFS, eigenvalue disconnect detection)  [O(N³) training]
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

from envs.gnm_env import (
    GNMEnv, GNMConfig,
    compute_node_features, compute_graph_features, compute_R,
    algebraic_connectivity,
    K_STEPS,
)
from models.gnm_policy import EdgeSelectionPolicy, PolicyConfig
import subprocess


def train(
    min_n: int = 8,
    max_n: int = 32,
    num_episodes: int = 10000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef_start: float = 0.1,
    entropy_coef_end: float = 0.01,
    clip_eps: float = 0.2,
    gae_lambda: float = 0.95,
    ppo_epochs: int = 4,
    device: str = "cpu",
    seed: int = 42,
    resume: str = None,
):
    """Train GNN edge selection policy with PPO."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"train_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    gnm_config = GNMConfig(min_n=min_n, max_n=max_n)
    policy_config = PolicyConfig(hidden_dim=64, num_gnn_layers=3)

    env = GNMEnv(gnm_config, seed=seed)
    policy = EdgeSelectionPolicy(policy_config).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=lr * 0.1)

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'policy_state_dict' in ckpt:
            policy.load_state_dict(ckpt['policy_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"Resumed from {resume} (episode {ckpt.get('episode', '?')})")
        else:
            policy.load_state_dict(ckpt)
            print(f"Resumed weights from {resume}")

    print("=" * 70)
    print("Training: PPO + GNN Edge Selection (3-layer message passing)")
    print("=" * 70)
    print(f"Graph size: n in [{min_n}, {max_n}]")
    print(f"Steps: K={K_STEPS} per episode, R=m//4 swaps per step")
    print(f"Architecture: GNN(3-layer D⁻¹A) + edge scorers")
    print(f"Episodes: {num_episodes}")
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"PPO: gamma={gamma}, clip_eps={clip_eps}, gae_lambda={gae_lambda}, epochs={ppo_epochs}")
    print(f"Entropy: {entropy_coef_start} -> {entropy_coef_end}")
    print(f"Node features: 3-dim (degree + 2-hop + 3-hop)")
    print("=" * 70)

    metrics_history = []
    best_avg_improvement = -float('inf')
    t_start = time.time()

    pbar = tqdm(range(num_episodes), desc="Training")

    for episode in pbar:
        progress = episode / max(num_episodes - 1, 1)
        entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress

        # Sample (n, m)
        n_sample = random.randint(min_n, max_n)
        max_possible_m = n_sample * (n_sample - 1) // 2
        min_m = n_sample - 1
        m_sample = random.randint(min_m, max_possible_m)

        state = env.reset(n=n_sample, m=m_sample)
        n = state.n
        m = state.m
        density = m / (n * (n - 1) / 2)
        R = compute_R(n, m)

        # Collect trajectory over K steps
        traj_node_feat = []
        traj_adj = []
        traj_edges = []
        traj_non_edges = []
        traj_graph_feat = []
        traj_remove_idx = []
        traj_add_idx = []
        traj_log_probs = []
        traj_values = []
        traj_rewards = []
        total_swaps = 0

        for k in range(K_STEPS):
            step_frac = k / K_STEPS

            # 1. Compute features — O(N²), no Lanczos
            node_feat = compute_node_features(env.adj, env.degrees, n)
            edges = env.get_edge_index()        # (M, 2)
            non_edges = env.get_non_edge_index() # (NE, 2)
            graph_feat = compute_graph_features(n, m, step_frac, max_n)  # (3,)

            # 2. Convert to tensors
            nf_t = torch.tensor(node_feat, dtype=torch.float32, device=device)
            adj_t = torch.tensor(env.adj, dtype=torch.float32, device=device)
            edges_t = torch.tensor(edges, dtype=torch.long, device=device)
            ne_t = torch.tensor(non_edges, dtype=torch.long, device=device)
            gf_t = torch.tensor(graph_feat, dtype=torch.float32, device=device)

            # 3. Policy forward → GNN encode + score + sample
            with torch.no_grad():
                remove_idx, add_idx, log_prob, value = policy.act(
                    nf_t, adj_t, edges_t, ne_t, gf_t, R
                )

            actual_R = len(remove_idx)

            # 4. Apply selected swaps
            if actual_R > 0:
                rem_edges_list = [(int(edges[i, 0]), int(edges[i, 1])) for i in remove_idx.cpu()]
                add_edges_list = [(int(non_edges[i, 0]), int(non_edges[i, 1])) for i in add_idx.cpu()]
                reward, info = env.spectral_step(rem_edges_list, add_edges_list)
                total_swaps += info['successful_swaps']
            else:
                reward = 0.0

            # Store trajectory
            traj_node_feat.append(node_feat)
            traj_adj.append(env.adj.copy())  # snapshot adj BEFORE swap applied
            traj_edges.append(edges)
            traj_non_edges.append(non_edges)
            traj_graph_feat.append(graph_feat)
            traj_remove_idx.append(remove_idx.cpu())
            traj_add_idx.append(add_idx.cpu())
            traj_log_probs.append(log_prob.item())
            traj_values.append(value.item())
            traj_rewards.append(reward)

        if len(traj_rewards) == 0:
            continue

        # Terminal λ₂ (exact)
        final_lambda2 = algebraic_connectivity(env.adj)
        terminal_improvement = final_lambda2 - env.initial_lambda2

        # --- GAE advantage computation ---
        values = traj_values
        rewards = traj_rewards

        advantages = []
        gae = 0.0
        for i in reversed(range(len(values))):
            next_val = values[i + 1] if i + 1 < len(values) else 0.0
            delta = rewards[i] + gamma * next_val - values[i]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values)]

        # Tensorize
        adv_batch = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_batch = torch.tensor(returns, dtype=torch.float32, device=device)
        old_lp_batch = torch.tensor(traj_log_probs, dtype=torch.float32, device=device)

        if len(adv_batch) > 1:
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        # --- PPO update ---
        for _ in range(ppo_epochs):
            new_lps = []
            new_ents = []
            new_vals = []

            for k in range(len(traj_graph_feat)):
                nf_t = torch.tensor(traj_node_feat[k], dtype=torch.float32, device=device)
                adj_t = torch.tensor(traj_adj[k], dtype=torch.float32, device=device)
                edges_t = torch.tensor(traj_edges[k], dtype=torch.long, device=device)
                ne_t = torch.tensor(traj_non_edges[k], dtype=torch.long, device=device)
                gf_t = torch.tensor(traj_graph_feat[k], dtype=torch.float32, device=device)
                rem_idx = traj_remove_idx[k].to(device)
                add_idx_t = traj_add_idx[k].to(device)

                lp, ent, val = policy.evaluate(
                    nf_t, adj_t, edges_t, ne_t, gf_t, rem_idx, add_idx_t
                )
                new_lps.append(lp)
                new_ents.append(ent)
                new_vals.append(val)

            new_lp = torch.stack(new_lps)
            new_ent = torch.stack(new_ents)
            new_val = torch.stack(new_vals)

            ratio = torch.exp(new_lp - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_val, ret_batch)
            entropy_loss = -new_ent.mean()

            loss = policy_loss + 0.5 * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()

        # Metrics
        metrics = {
            'episode': episode,
            'lambda2': final_lambda2,
            'improvement': terminal_improvement,
            'mean_reward': np.mean(traj_rewards),
            'R': R,
            'total_swaps': total_swaps,
            'entropy': new_ent.mean().item(),
            'n': n,
            'm': m,
            'density': density,
        }
        metrics_history.append(metrics)

        pbar.set_postfix({
            'l2': f"{final_lambda2:.2f}",
            'dl2': f"{terminal_improvement:.3f}",
            'R': R,
            'sw': total_swaps,
            'n': n,
            'd': f"{density:.2f}",
        })

        if (episode + 1) % 100 == 0:
            recent = metrics_history[-100:]
            avg_improvement = np.mean([m['improvement'] for m in recent])

            if avg_improvement > best_avg_improvement:
                best_avg_improvement = avg_improvement
                torch.save(policy.state_dict(), log_dir / "best_policy.pt")

        # Adaptive checkpoint frequency
        ckpt_freq = 5000 if num_episodes >= 50000 else 1000
        report_freq = ckpt_freq

        if (episode + 1) % report_freq == 0:
            recent = metrics_history[-report_freq:]
            avg_lambda2 = np.mean([m['lambda2'] for m in recent])
            avg_improvement = np.mean([m['improvement'] for m in recent])
            avg_R = np.mean([m['R'] for m in recent])
            avg_swaps = np.mean([m['total_swaps'] for m in recent])
            elapsed = time.time() - t_start
            eps_per_sec = (episode + 1) / elapsed
            eta_sec = (num_episodes - episode - 1) / max(eps_per_sec, 0.01)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\n--- Episode {episode + 1}/{num_episodes} [{elapsed/60:.1f}m elapsed, ETA {eta_sec/60:.1f}m] ---")
            print(f"  Avg l2: {avg_lambda2:.4f}  |  Avg dl2: {avg_improvement:.4f}  |  Best dl2: {best_avg_improvement:.4f}")
            print(f"  Avg R: {avg_R:.2f}  |  Avg swaps: {avg_swaps:.1f}  |  LR: {current_lr:.2e}  |  {eps_per_sec:.1f} ep/s")

            checkpoint_path = log_dir / f"checkpoint_{episode + 1}.pt"
            torch.save({
                'episode': episode + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'hidden_dim': policy_config.hidden_dim,
                    'num_gnn_layers': policy_config.num_gnn_layers,
                },
                'metrics': {
                    'avg_lambda2': avg_lambda2,
                    'avg_improvement': avg_improvement,
                    'best_avg_improvement': best_avg_improvement,
                    'episodes_per_sec': eps_per_sec,
                    'elapsed_minutes': elapsed / 60,
                }
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path.name}")

    total_time = time.time() - t_start
    torch.save(policy.state_dict(), log_dir / "final_policy.pt")
    with open(log_dir / "metrics.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"  Episodes: {num_episodes}  |  Time: {total_time/60:.1f}m  |  {num_episodes/total_time:.1f} ep/s")
    print(f"  Best avg improvement: {best_avg_improvement:.4f}")
    print(f"  Saved to: {log_dir}")
    print(f"{'='*70}")

    return log_dir, metrics_history


def evaluate_via_baselines(checkpoint_path: str, n_values: list = None):
    """Compare to baselines via eval_checkpoint.py."""
    if n_values is None:
        n_values = [16, 24, 32]
    n_str = ",".join(str(n) for n in n_values)
    eval_script = Path(__file__).parent / "eval_checkpoint.py"
    cmd = [
        "python3", str(eval_script), str(checkpoint_path),
        "--n", n_str, "--trials", "5", "--workers", "4",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Train G(n,m) Model (PPO + GNN Edge Selection)")
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef-start", type=float, default=0.1)
    parser.add_argument("--entropy-coef-end", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, default=None, help="Evaluate existing policy")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if args.eval:
        evaluate_via_baselines(args.eval, n_values=[16, 24, 32])
    else:
        log_dir, _ = train(
            min_n=args.min_n,
            max_n=args.max_n,
            num_episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            clip_eps=args.clip_eps,
            gae_lambda=args.gae_lambda,
            ppo_epochs=args.ppo_epochs,
            entropy_coef_start=args.entropy_coef_start,
            entropy_coef_end=args.entropy_coef_end,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
        )
        policy_file = log_dir / "best_policy.pt"
        if not policy_file.exists():
            policy_file = log_dir / "final_policy.pt"

        print("\nCompare to baselines via eval_checkpoint.py...")
        evaluate_via_baselines(str(policy_file), n_values=[16, 24, 32])


if __name__ == "__main__":
    main()
