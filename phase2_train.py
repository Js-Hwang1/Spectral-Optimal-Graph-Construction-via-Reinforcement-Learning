"""
Phase 2 Training: Production-Grade RL for Residual Edge Placement

This module implements a research-grade training pipeline for learning to place
edges that maximize algebraic connectivity (lambda_2).

Key Features:
1. Curriculum Learning: Start with small graphs, gradually increase complexity
2. PPO with modern tricks: Advantage normalization, entropy bonus, gradient clipping
3. Multi-step returns for stable learning
4. Validation during training with diverse graph configurations
5. Proper handling of variable-size graphs through size-stratified batching
6. Comprehensive logging and checkpointing

Training Objective:
- Learn a policy that places residual edges to maximize lambda_2
- The policy must generalize across different (n, m') configurations
- Inference must be O(N^2) without eigenvalue computation
"""

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time
import json
import argparse
import logging

from phase1 import AdaptiveGraphBuilder
from phase2 import EdgePolicyNetwork, GraphFeatureExtractor


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class TrainingConfig:
    """Training hyperparameters with sensible defaults for graph RL."""

    # Environment
    min_n: int = 15
    max_n: int = 60
    min_density: float = 0.15
    max_density: float = 0.50
    skeleton_ratio: float = 0.7

    # Curriculum learning (disabled by default)
    curriculum_enabled: bool = False
    curriculum_stages: int = 1
    episodes_per_stage: int = 20000

    # Model architecture
    hidden_dim: int = 128
    num_gat_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2  # Value function clipping
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_decay: float = 0.9995  # Decay entropy over time
    max_grad_norm: float = 0.5

    # Training
    lr: float = 3e-4
    lr_min: float = 1e-5
    batch_size: int = 32
    n_epochs: int = 4
    rollout_steps: int = 2048  # Steps before each update
    num_episodes: int = 20000

    # Validation & logging
    val_freq: int = 500
    log_freq: int = 100
    save_freq: int = 1000

    # Reward shaping
    lambda2_reward_scale: float = 1.0
    step_penalty: float = 0.0  # Small penalty per step to encourage efficiency
    completion_bonus: float = 0.5

    # Paths
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    seed: int = 42
    device: str = "auto"


# =============================================================================
# 2. GRAPH ENVIRONMENT (Enhanced)
# =============================================================================
class GraphEnv:
    """
    RL Environment for residual edge placement on Phase 1 skeletons.

    Training Flow:
    1. reset() builds a Phase 1 skeleton using AdaptiveGraphBuilder
    2. Agent starts with skeleton as initial state
    3. Agent adds residual edges one-by-one via step()
    4. Episode ends when all residual edges are placed

    This ensures the agent learns to optimize ONLY the residual edge
    placement, not the entire graph construction.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.feature_extractor = GraphFeatureExtractor()

        # Environment bounds
        self.n_min = config.min_n
        self.n_max = config.max_n
        self.density_min = config.min_density
        self.density_max = config.max_density

        # State
        self.n = None
        self.adj = None
        self.target_edges = None
        self.current_edges = None
        self.edges_remaining = None
        self.prev_lambda2 = None
        self.step_count = 0

    def reset(self, n: int = None, m: int = None) -> Dict[str, np.ndarray]:
        """
        Reset environment with a new graph.

        Args:
            n: Optional fixed node count (for validation)
            m: Optional fixed edge count (for validation)
        """
        # Sample or use provided graph size
        if n is None:
            self.n = self.rng.integers(self.n_min, self.n_max + 1)
        else:
            self.n = n

        max_edges = self.n * (self.n - 1) // 2

        if m is None:
            density = self.rng.uniform(self.density_min, self.density_max)
            self.target_edges = int(max_edges * density)
        else:
            self.target_edges = min(m, max_edges)

        # =================================================================
        # PHASE 1: Build deterministic skeleton (agent does NOT control this)
        # =================================================================
        # Ensure we always have at least a few edges to place in Phase 2
        min_residual = max(3, self.n // 5)  # At least 3 edges or n/5
        max_skeleton = self.target_edges - min_residual
        skeleton_edges = int(self.target_edges * self.config.skeleton_ratio)
        skeleton_edges = max(self.n - 1, min(skeleton_edges, max_skeleton))

        # If target is too small, increase it
        if skeleton_edges >= self.target_edges:
            self.target_edges = skeleton_edges + min_residual

        # Build Phase 1 skeleton - this is the starting point for the RL agent
        builder = AdaptiveGraphBuilder(self.n)
        builder.build(skeleton_edges)

        # Agent's initial state: the Phase 1 skeleton
        self.adj = builder.adj.astype(np.float32)
        self.current_edges = len(builder.edges)

        # =================================================================
        # PHASE 2: Agent will add these residual edges
        # =================================================================
        self.edges_remaining = self.target_edges - self.current_edges
        self.step_count = 0

        # Initial lambda_2 of the skeleton (before agent adds any edges)
        self.prev_lambda2 = self._compute_lambda2()

        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Add a residual edge to the skeleton.

        The agent selects which edge to add from the remaining candidates
        (edges not already in the skeleton). This is Phase 2 of the pipeline.
        """
        i, j = self._flat_to_pair(action)
        self.step_count += 1

        # Validate action
        if i >= self.n or j >= self.n or self.adj[i, j] != 0:
            return self._get_obs(), -0.5, False, {'invalid': True, 'lambda2': self.prev_lambda2}

        # Add edge
        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self.current_edges += 1
        self.edges_remaining -= 1

        # Compute new lambda_2
        new_lambda2 = self._compute_lambda2()

        # Reward shaping
        lambda2_improvement = new_lambda2 - self.prev_lambda2
        reward = self.config.lambda2_reward_scale * lambda2_improvement

        # Step penalty (encourages efficient edge use)
        reward -= self.config.step_penalty

        self.prev_lambda2 = new_lambda2

        # Termination
        done = self.edges_remaining <= 0

        if done:
            # Completion bonus proportional to final lambda_2
            reward += self.config.completion_bonus * new_lambda2 / self.n

        info = {
            'lambda2': new_lambda2,
            'lambda2_improvement': lambda2_improvement,
            'edges_placed': self.current_edges,
            'edges_remaining': self.edges_remaining,
            'n': self.n,
            'step': self.step_count,
            'invalid': False
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation with all features."""
        degrees = self.adj.sum(axis=1)
        node_features = self.feature_extractor.get_node_features(self.adj)
        edge_features = self.feature_extractor.get_edge_features(self.adj, node_features)
        global_features = self.feature_extractor.get_global_features(self.adj, degrees)

        # Candidate mask
        num_edges = self.n * (self.n - 1) // 2
        candidate_mask = np.zeros(num_edges, dtype=bool)
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                candidate_mask[idx] = (self.adj[i, j] == 0)
                idx += 1

        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'global_features': global_features,
            'adj': self.adj.copy(),
            'candidate_mask': candidate_mask,
            'n': self.n,
            'edges_remaining': self.edges_remaining
        }

    def _compute_lambda2(self) -> float:
        """Compute algebraic connectivity using exact solver."""
        degrees = np.sum(self.adj, axis=0)
        L = np.diag(degrees) - self.adj
        try:
            eigvals = scipy.linalg.eigh(L, eigvals_only=True)
            eigvals.sort()
            return eigvals[1] if len(eigvals) > 1 else 0.0
        except:
            return 0.0

    def _flat_to_pair(self, flat_idx: int) -> Tuple[int, int]:
        """Convert flat index to (i, j) pair."""
        n = self.n
        i = int(n - 2 - np.floor(np.sqrt(-8 * flat_idx + 4 * n * (n - 1) - 7) / 2 - 0.5))
        j = int(flat_idx + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
        return i, j


# =============================================================================
# 3. EXPERIENCE STORAGE WITH GAE
# =============================================================================
@dataclass
class Transition:
    """Single step transition."""
    node_features: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    adj: np.ndarray
    candidate_mask: np.ndarray
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    n: int


class RolloutBuffer:
    """
    Buffer for storing rollout experiences with GAE computation.

    Handles variable-size graphs by grouping by size for efficient batching.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.transitions: List[Transition] = []
        self.episode_starts: List[int] = [0]

    def add(self, transition: Transition):
        self.transitions.append(transition)

    def mark_episode_end(self):
        """Mark the end of an episode for proper GAE computation."""
        if len(self.transitions) > 0:
            self.episode_starts.append(len(self.transitions))

    def clear(self):
        self.transitions = []
        self.episode_starts = [0]

    def compute_returns_and_advantages(self, last_values: Dict[int, float] = None):
        """
        Compute GAE advantages and returns.

        Handles episode boundaries properly for multi-episode buffers.
        """
        if last_values is None:
            last_values = {}

        n_transitions = len(self.transitions)
        advantages = np.zeros(n_transitions, dtype=np.float32)
        returns = np.zeros(n_transitions, dtype=np.float32)

        # Process each episode separately
        for ep_idx in range(len(self.episode_starts) - 1):
            start_idx = self.episode_starts[ep_idx]
            end_idx = self.episode_starts[ep_idx + 1]

            if start_idx >= end_idx:
                continue

            gae = 0
            last_t = self.transitions[end_idx - 1]
            next_value = last_values.get(last_t.n, 0.0) if not last_t.done else 0.0

            for t in range(end_idx - 1, start_idx - 1, -1):
                trans = self.transitions[t]
                next_non_terminal = 0.0 if trans.done else 1.0

                delta = trans.reward + self.gamma * next_value * next_non_terminal - trans.value
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

                advantages[t] = gae
                returns[t] = gae + trans.value
                next_value = trans.value

        # Handle last partial episode if exists
        if self.episode_starts[-1] < n_transitions:
            start_idx = self.episode_starts[-1]
            gae = 0
            last_t = self.transitions[-1]
            next_value = last_values.get(last_t.n, 0.0) if not last_t.done else 0.0

            for t in range(n_transitions - 1, start_idx - 1, -1):
                trans = self.transitions[t]
                next_non_terminal = 0.0 if trans.done else 1.0

                delta = trans.reward + self.gamma * next_value * next_non_terminal - trans.value
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

                advantages[t] = gae
                returns[t] = gae + trans.value
                next_value = trans.value

        return advantages, returns

    def get_batches(self, batch_size: int, device: str, shuffle: bool = True):
        """
        Generate batches grouped by graph size.

        Returns list of batches, each containing same-size graphs.
        """
        advantages, returns = self.compute_returns_and_advantages()

        # Group by graph size
        size_groups: Dict[int, List[int]] = {}
        for i, trans in enumerate(self.transitions):
            n = trans.n
            if n not in size_groups:
                size_groups[n] = []
            size_groups[n].append(i)

        # Create batches per size
        batches = []
        for n, indices in size_groups.items():
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = self._create_batch(batch_indices, n, advantages, returns, device)
                batches.append(batch)

        if shuffle:
            np.random.shuffle(batches)

        return batches

    def _create_batch(self, indices: List[int], n: int,
                      advantages: np.ndarray, returns: np.ndarray,
                      device: str) -> Dict[str, torch.Tensor]:
        """Create tensor batch from indices."""
        batch_size = len(indices)
        num_edges = n * (n - 1) // 2

        # Pre-allocate tensors
        node_features = torch.zeros(batch_size, n, 8, device=device)
        edge_features = torch.zeros(batch_size, num_edges, 10, device=device)
        global_features = torch.zeros(batch_size, 6, device=device)
        adj = torch.zeros(batch_size, n, n, device=device)
        candidate_mask = torch.zeros(batch_size, num_edges, dtype=torch.bool, device=device)
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        old_log_probs = torch.zeros(batch_size, device=device)
        old_values = torch.zeros(batch_size, device=device)
        advs = torch.zeros(batch_size, device=device)
        rets = torch.zeros(batch_size, device=device)

        for i, idx in enumerate(indices):
            trans = self.transitions[idx]
            node_features[i] = torch.from_numpy(trans.node_features)
            edge_features[i] = torch.from_numpy(trans.edge_features)
            global_features[i] = torch.from_numpy(trans.global_features)
            adj[i] = torch.from_numpy(trans.adj)
            candidate_mask[i] = torch.from_numpy(trans.candidate_mask)
            actions[i] = trans.action
            old_log_probs[i] = trans.log_prob
            old_values[i] = trans.value
            advs[i] = float(advantages[idx])
            rets[i] = float(returns[idx])

        # Normalize advantages within batch
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'global_features': global_features,
            'adj': adj,
            'candidate_mask': candidate_mask,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
            'advantages': advs,
            'returns': rets,
            'n': n
        }


# =============================================================================
# 4. PPO TRAINER (Production-Grade)
# =============================================================================
class PPOTrainer:
    """
    Proximal Policy Optimization with all modern improvements.

    Features:
    - Clipped objective for both policy and value
    - Entropy bonus with decay
    - Gradient clipping
    - Learning rate scheduling
    - Proper advantage normalization
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device(config.device)

        # Model
        self.model = EdgePolicyNetwork(
            node_feat_dim=8,
            edge_feat_dim=10,
            global_feat_dim=6,
            hidden_dim=config.hidden_dim,
            num_gat_layers=config.num_gat_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        ).to(self.device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=1e-5,
            eps=1e-5
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.episodes_per_stage,
            T_mult=2,
            eta_min=config.lr_min
        )

        # Buffer
        self.buffer = RolloutBuffer(gamma=config.gamma, gae_lambda=config.gae_lambda)

        # Entropy coefficient (will decay)
        self.ent_coef = config.ent_coef

        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0

    def _get_device(self, device_str: str) -> str:
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str

    def select_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, float, float]:
        """Select action using current policy."""
        with torch.no_grad():
            # Handle NaN in input features
            node_feat = torch.tensor(obs['node_features'], device=self.device, dtype=torch.float32).unsqueeze(0)
            edge_feat = torch.tensor(obs['edge_features'], device=self.device, dtype=torch.float32).unsqueeze(0)
            global_feat = torch.tensor(obs['global_features'], device=self.device, dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(obs['adj'], device=self.device, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(obs['candidate_mask'], device=self.device).unsqueeze(0)

            # Replace NaN with 0 in features
            node_feat = torch.nan_to_num(node_feat, nan=0.0)
            edge_feat = torch.nan_to_num(edge_feat, nan=0.0)
            global_feat = torch.nan_to_num(global_feat, nan=0.0)

            # Check if any valid actions exist
            if not mask.any():
                # No valid actions - return random invalid action (will be penalized)
                return 0, 0.0, 0.0

            edge_logits, value = self.model(node_feat, edge_feat, adj, global_feat, mask)

            # Handle NaN/Inf in logits
            edge_logits = torch.nan_to_num(edge_logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

            # Compute probabilities with numerical stability
            # Only compute over valid actions
            valid_logits = edge_logits.clone()
            valid_logits[~mask] = -1e9

            probs = F.softmax(valid_logits, dim=-1)

            # Ensure valid probability distribution
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                # Sample from categorical
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            log_prob = F.log_softmax(valid_logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.nan_to_num(log_prob, nan=0.0)

        return action.item(), log_prob.item(), value.item()

    def train_step(self) -> Dict[str, float]:
        """Perform PPO update on collected experiences."""
        if len(self.buffer.transitions) == 0:
            return {}

        self.model.train()

        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []

        for epoch in range(self.config.n_epochs):
            batches = self.buffer.get_batches(self.config.batch_size, self.device)

            for batch in batches:
                # Forward pass
                edge_logits, values = self.model(
                    batch['node_features'],
                    batch['edge_features'],
                    batch['adj'],
                    batch['global_features'],
                    batch['candidate_mask']
                )

                # New log probs and entropy
                log_probs = F.log_softmax(edge_logits, dim=-1)
                probs = F.softmax(edge_logits, dim=-1)
                new_log_probs = log_probs.gather(1, batch['actions'].unsqueeze(-1)).squeeze(-1)

                # Entropy (only over valid actions)
                valid_probs = probs.masked_fill(~batch['candidate_mask'], 0)
                valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                entropy = -(valid_probs * torch.log(valid_probs + 1e-8)).sum(dim=-1).mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss
                values_pred = values.squeeze(-1)
                values_clipped = batch['old_values'] + torch.clamp(
                    values_pred - batch['old_values'],
                    -self.config.clip_range_vf,
                    self.config.clip_range_vf
                )
                value_loss_unclipped = (values_pred - batch['returns']) ** 2
                value_loss_clipped = (values_clipped - batch['returns']) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Total loss
                loss = (policy_loss +
                        self.config.vf_coef * value_loss -
                        self.ent_coef * entropy)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

                with torch.no_grad():
                    approx_kl = (batch['old_log_probs'] - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.config.clip_range).float().mean().item()
                    approx_kls.append(approx_kl)
                    clip_fractions.append(clip_frac)

        # Decay entropy coefficient
        self.ent_coef *= self.config.ent_coef_decay
        self.ent_coef = max(self.ent_coef, 1e-4)

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
            'ent_coef': self.ent_coef
        }

    def save(self, path: str, extra_info: Dict = None):
        """Save checkpoint with full training state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hidden_dim': self.config.hidden_dim,
            'node_feat_dim': 8,
            'edge_feat_dim': 10,
            'num_gat_layers': self.config.num_gat_layers,
            'num_heads': self.config.num_heads,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'ent_coef': self.ent_coef,
        }
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        if 'total_episodes' in checkpoint:
            self.total_episodes = checkpoint['total_episodes']
        if 'ent_coef' in checkpoint:
            self.ent_coef = checkpoint['ent_coef']


# =============================================================================
# 5. VALIDATION
# =============================================================================
class Validator:
    """Validate trained model on diverse graph configurations."""

    def __init__(self, trainer: PPOTrainer, config: TrainingConfig):
        self.trainer = trainer
        self.config = config

        # Fixed validation configurations
        self.val_configs = [
            (15, 30), (15, 60),
            (20, 50), (20, 100),
            (30, 80), (30, 160),
            (40, 120), (40, 250),
            (50, 180), (50, 400),
        ]

    def validate(self) -> Dict[str, float]:
        """Run validation on fixed configurations."""
        self.trainer.model.eval()

        results = []
        for n, m in self.val_configs:
            env = GraphEnv(self.config)
            obs = env.reset(n=n, m=m)

            initial_lambda2 = env.prev_lambda2
            episode_reward = 0
            done = False

            while not done and env.edges_remaining > 0:
                action, _, _ = self.trainer.select_action(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            final_lambda2 = info['lambda2']
            improvement = final_lambda2 - initial_lambda2

            results.append({
                'n': n,
                'm': m,
                'initial_lambda2': initial_lambda2,
                'final_lambda2': final_lambda2,
                'improvement': improvement,
                'reward': episode_reward
            })

        # Aggregate metrics
        mean_improvement = np.mean([r['improvement'] for r in results])
        mean_final_lambda2 = np.mean([r['final_lambda2'] for r in results])
        mean_reward = np.mean([r['reward'] for r in results])

        return {
            'val_mean_improvement': mean_improvement,
            'val_mean_lambda2': mean_final_lambda2,
            'val_mean_reward': mean_reward,
            'val_results': results
        }


# =============================================================================
# 6. MAIN TRAINING LOOP
# =============================================================================
def train(config: TrainingConfig):
    """Main training loop with curriculum learning."""
    # Create directories first
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Initialize
    trainer = PPOTrainer(config)
    validator = Validator(trainer, config)

    logger.info(f"Training on device: {trainer.device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Metrics tracking
    episode_rewards = deque(maxlen=100)
    episode_lambda2s = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_val_lambda2 = -float('inf')

    # Training loop
    env = GraphEnv(config)

    for episode in range(config.num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        info = {'lambda2': env.prev_lambda2}  # Initialize with current lambda2

        while not done and env.edges_remaining > 0:
            # Select action
            action, log_prob, value = trainer.select_action(obs)

            # Take step
            next_obs, reward, done, info = env.step(action)

            # Store transition
            transition = Transition(
                node_features=obs['node_features'],
                edge_features=obs['edge_features'],
                global_features=obs['global_features'],
                adj=obs['adj'],
                candidate_mask=obs['candidate_mask'],
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
                n=obs['n']
            )
            trainer.buffer.add(transition)

            episode_reward += reward
            episode_steps += 1
            trainer.total_steps += 1
            obs = next_obs

        trainer.buffer.mark_episode_end()
        trainer.total_episodes += 1

        # Episode metrics
        final_lambda2 = info['lambda2']
        episode_rewards.append(episode_reward)
        episode_lambda2s.append(final_lambda2)
        episode_lengths.append(episode_steps)

        # Train when buffer is full enough
        if len(trainer.buffer.transitions) >= config.rollout_steps:
            train_metrics = trainer.train_step()
            trainer.scheduler.step()

        # Logging
        if (episode + 1) % config.log_freq == 0:
            avg_reward = np.mean(episode_rewards)
            avg_lambda2 = np.mean(episode_lambda2s)
            avg_length = np.mean(episode_lengths)

            logger.info(
                f"Ep {episode + 1}/{config.num_episodes} | "
                f"Reward: {avg_reward:.3f} | "
                f"L2: {avg_lambda2:.4f} | "
                f"Len: {avg_length:.1f} | "
                f"N: [{env.n_min}-{env.n_max}]"
            )

        # Validation
        if (episode + 1) % config.val_freq == 0:
            val_metrics = validator.validate()
            logger.info(
                f"  [VAL] Mean L2: {val_metrics['val_mean_lambda2']:.4f} | "
                f"Improvement: {val_metrics['val_mean_improvement']:.4f}"
            )

            if val_metrics['val_mean_lambda2'] > best_val_lambda2:
                best_val_lambda2 = val_metrics['val_mean_lambda2']
                trainer.save(
                    save_dir / 'best_model.pt',
                    {'best_val_lambda2': best_val_lambda2, 'episode': episode + 1}
                )
                logger.info(f"  [NEW BEST] Saved best model (L2={best_val_lambda2:.4f})")

        # Checkpointing
        if (episode + 1) % config.save_freq == 0:
            trainer.save(save_dir / f'checkpoint_{episode + 1}.pt')

    # Final save
    trainer.save(save_dir / 'final_model.pt')
    logger.info(f"Training complete. Best validation L2: {best_val_lambda2:.4f}")


# =============================================================================
# 7. EVALUATION
# =============================================================================
def evaluate(model_path: str, device: str = "auto"):
    """Comprehensive evaluation of trained model."""
    from phase2 import ResidualEdgeFiller

    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"

    print(f"Evaluating model: {model_path}")
    print("=" * 80)

    # Test configurations
    test_configs = [
        (15, 25), (15, 40), (15, 60),
        (20, 40), (20, 80), (20, 120),
        (30, 70), (30, 140), (30, 220),
        (40, 100), (40, 200), (40, 350),
        (50, 150), (50, 300), (50, 500),
        (60, 200), (60, 400), (60, 700),
    ]

    # Load model
    filler = ResidualEdgeFiller(model_path=model_path, device=device)

    results = []
    print(f"{'N':>4} {'M':>5} {'Phase1':>10} {'Heuristic':>10} {'Model':>10} {'Best':>10}")
    print("-" * 60)

    for n, m in test_configs:
        # Phase 1 baseline
        m_skeleton = int(m * 0.7)
        builder = AdaptiveGraphBuilder(n)
        builder.build(m_skeleton)
        phase1_lambda2 = builder.get_lambda2()

        # Heuristic
        m_rem = m - len(builder.edges)
        filler_heur = ResidualEdgeFiller()
        adj_heur = filler_heur.fill_edges(builder.adj.astype(np.float32), m_rem)
        heur_lambda2 = _compute_lambda2(adj_heur)

        # Model
        if filler.model is not None:
            adj_model = filler.fill_edges(builder.adj.astype(np.float32), m_rem)
            model_lambda2 = _compute_lambda2(adj_model)
        else:
            model_lambda2 = heur_lambda2

        best = max(phase1_lambda2, heur_lambda2, model_lambda2)
        winner = "Phase1" if best == phase1_lambda2 else ("Heur" if best == heur_lambda2 else "Model")

        print(f"{n:>4} {m:>5} {phase1_lambda2:>10.4f} {heur_lambda2:>10.4f} {model_lambda2:>10.4f} {winner:>10}")

        results.append({
            'n': n, 'm': m,
            'phase1': phase1_lambda2,
            'heuristic': heur_lambda2,
            'model': model_lambda2,
            'winner': winner
        })

    print("-" * 60)

    # Summary
    model_wins = sum(1 for r in results if r['winner'] == 'Model')
    print(f"\nModel wins: {model_wins}/{len(results)}")
    print(f"Avg model L2: {np.mean([r['model'] for r in results]):.4f}")
    print(f"Avg improvement over Phase1: {np.mean([r['model'] - r['phase1'] for r in results]):.4f}")


def _compute_lambda2(adj: np.ndarray) -> float:
    """Helper to compute lambda2."""
    degrees = np.sum(adj, axis=0)
    L = np.diag(degrees) - adj
    eigvals = scipy.linalg.eigh(L, eigvals_only=True)
    eigvals.sort()
    return eigvals[1] if len(eigvals) > 1 else 0.0


# =============================================================================
# 8. CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 RL Training for Maximum Algebraic Connectivity'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--num-episodes', type=int, default=20000)
    train_parser.add_argument('--min-n', type=int, default=15)
    train_parser.add_argument('--max-n', type=int, default=60)
    train_parser.add_argument('--hidden-dim', type=int, default=128)
    train_parser.add_argument('--num-gat-layers', type=int, default=3)
    train_parser.add_argument('--lr', type=float, default=3e-4)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--save-dir', type=str, default='./checkpoints')
    train_parser.add_argument('--log-dir', type=str, default='./logs')
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--device', type=str, default='auto')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--device', type=str, default='auto')

    # Quick train for testing
    quick_parser = subparsers.add_parser('quick', help='Quick training run for testing')
    quick_parser.add_argument('--episodes', type=int, default=500)
    quick_parser.add_argument('--save-dir', type=str, default='./quick_checkpoints')

    args = parser.parse_args()

    if args.command == 'train':
        config = TrainingConfig(
            num_episodes=args.num_episodes,
            min_n=args.min_n,
            max_n=args.max_n,
            hidden_dim=args.hidden_dim,
            num_gat_layers=args.num_gat_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            seed=args.seed,
            device=args.device
        )
        train(config)

    elif args.command == 'eval':
        evaluate(args.model, args.device)

    elif args.command == 'quick':
        config = TrainingConfig(
            num_episodes=args.episodes,
            val_freq=100,
            log_freq=50,
            save_freq=200,
            save_dir=args.save_dir,
            log_dir=args.save_dir,
            max_n=40
        )
        train(config)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
