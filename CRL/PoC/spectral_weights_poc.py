#!/usr/bin/env python3
"""
PoC: Learned Spectral Weights — RL learns to blend eigenvector directions.

=== Problem ===
FV uses score(i,j) = (v₂[i] - v₂[j])², the exact gradient of λ₂.
Greedy argmax on this gradient gets trapped in local optima (non-convex landscape).
Temperature PoC showed that deviating from greedy helps — but fixed noise is crude.

=== Idea ===
The Laplacian has eigenvectors v₂, v₃, ..., v_n, each carrying structural info.
FV ignores all but v₂. We learn a state-dependent blending:

  score(i,j) = Σ_k  w_k(state) · n · (v_k[i] - v_k[j])²

When w = [1, 0, 0, ...] → standard FV (only v₂).
The agent learns WHEN to use higher eigenvectors (the "direction" of deviation).

The weights w(state) depend on size-invariant spectral features:
  - Gap ratios: (λ_{k+1} - λ_k) / λ_k  (detects degeneracy)
  - Density: m / m_max
  - Progress: step / total_steps

=== Training ===
REINFORCE with per-step Δλ₂ rewards on n=8..10.
The weight network has ~350 params (tiny — should generalize).

=== Evaluation ===
Greedy argmax on blended scores. Compare to standard FV (same init, same seeds).
Test generalization at n=12, 16, 24.

=== Key questions ===
1. Can learned weights beat w=[1,0,...] (FV)?
2. Do they generalize to unseen n?
3. What patterns emerge? (When does the agent deviate from FV?)

Usage:
    python spectral_weights_poc.py
    python spectral_weights_poc.py --episodes 20000 --eval-n 8,10,12,16,24
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy import linalg
import time

# =====================================================================
# Constants
# =====================================================================
K_EIGVECS = 8                       # v₂ through v₉
STATE_DIM = (K_EIGVECS - 1) + 2     # 7 gap ratios + density + progress = 9
HIDDEN_DIM = 32
GAMMA = 0.99
TRAIN_N = [8, 9, 10]

# =====================================================================
# Baselines cache
# =====================================================================
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


# =====================================================================
# Graph utilities
# =====================================================================

def random_spanning_tree(n, rng):
    """Wilson's algorithm for uniform random spanning tree."""
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


# =====================================================================
# Weight Network
# =====================================================================

class WeightNet(nn.Module):
    """
    Maps spectral state → blending weights on the simplex.

    Input (9-dim): 7 spectral gap ratios + density + progress
    Output (8-dim): softmax weights for v₂ through v₉
    """

    def __init__(self, state_dim=STATE_DIM, k=K_EIGVECS, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, k),
        )
        # Initialize: v₂ dominant (approximates FV at start)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.copy_(
                torch.tensor([2.0] + [0.0] * (k - 1)))

    def forward(self, state):
        return torch.softmax(self.net(state), dim=-1)


# =====================================================================
# State features (size-invariant)
# =====================================================================

def compute_state(evals, current_m, max_m, step, total_steps):
    """
    9-dim state from eigenvalues + context.

    Features 0-6: gap ratios (λ_{k+1} - λ_k) / λ_k for k=2..8
                   Detects eigenvalue degeneracy (when FV gradient is unstable).
    Feature 7:     density = m_current / m_max
    Feature 8:     progress = step / total_steps
    """
    gaps = []
    for i in range(1, K_EIGVECS):  # i=1..7 → gaps between λ₂-λ₃, ..., λ₈-λ₉
        if i + 1 < len(evals):
            lam_k = max(abs(evals[i]), 1e-8)
            gap = (evals[i + 1] - evals[i]) / lam_k
            gaps.append(float(np.clip(gap, 0, 5.0)))
        else:
            gaps.append(0.0)

    density = current_m / max(max_m, 1)
    progress = step / max(total_steps, 1)

    return gaps + [density, progress]


# =====================================================================
# Episode collection
# =====================================================================

def collect_episode(n, m, weight_net, seed, tau=0.1, training=True):
    """
    Build graph from spanning tree to m edges using learned weights.

    At each step:
      1. Eigendecompose → eigenvalues + eigenvectors
      2. State → weight network → blending weights w_k
      3. score(i,j) = Σ_k w_k · n · (v_k[i] - v_k[j])²
      4. Select edge (softmax sample if training, argmax if eval)
      5. Reward = Δλ₂

    Returns: (final_λ₂, log_probs, rewards)
    """
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    if edges_to_add <= 0:
        return algebraic_connectivity(adj), [], []

    max_m = n * (n - 1) // 2
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    k_use = min(K_EIGVECS, n - 1)

    log_probs = []
    rewards = []
    current_m = n - 1

    for step in range(edges_to_add):
        # Full eigendecomposition
        L = np.diag(adj.sum(axis=1)) - adj
        evals, evecs = linalg.eigh(L)
        lam2_before = float(evals[1])

        # State → weights
        state_list = compute_state(evals, current_m, max_m, step, edges_to_add)
        state_t = torch.tensor(state_list, dtype=torch.float32)
        weights = weight_net(state_t)  # [K], simplex

        # Valid non-edges
        mask = (adj == 0) & upper_tri
        if not mask.any():
            break
        valid_idx = np.where(mask.ravel())[0]
        num_valid = len(valid_idx)

        # Per-eigenvector scores for valid edges (n-normalized for invariance)
        scores_per_k = np.zeros((K_EIGVECS, num_valid))
        for k in range(k_use):
            vk = evecs[:, k + 1]
            all_gaps = n * (vk[:, None] - vk[None, :]) ** 2
            scores_per_k[k] = all_gaps.ravel()[valid_idx]

        # Blend (differentiable through weights)
        scores_t = torch.from_numpy(scores_per_k).float()
        blended = (weights.unsqueeze(1) * scores_t).sum(dim=0)

        if training:
            log_softmax = torch.log_softmax(blended / tau, dim=0)
            probs_np = torch.softmax(blended / tau, dim=0).detach().numpy()
            idx = rng.choice(num_valid, p=probs_np)
            log_probs.append(log_softmax[idx])
        else:
            idx = blended.argmax().item()

        chosen = valid_idx[idx]
        i, j = chosen // n, chosen % n
        adj[i, j] = adj[j, i] = 1
        current_m += 1

        # Per-step reward: exact Δλ₂
        lam2_after = float(linalg.eigvalsh(
            np.diag(adj.sum(axis=1)) - adj)[1])
        rewards.append(lam2_after - lam2_before)

    return algebraic_connectivity(adj), log_probs, rewards


def fv_construct(n, m, seed):
    """Standard FV (greedy v₂ gap) for controlled comparison."""
    rng = np.random.default_rng(seed)
    adj = random_spanning_tree(n, rng)
    edges_to_add = m - (n - 1)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    for _ in range(edges_to_add):
        L = np.diag(adj.sum(axis=1)) - adj
        _, evecs = linalg.eigh(L)
        v2 = evecs[:, 1]
        scores = (v2[:, None] - v2[None, :]) ** 2

        mask = (adj == 0) & upper_tri
        if not mask.any():
            break

        scores[~mask] = -1
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        adj[idx[0], idx[1]] = adj[idx[1], idx[0]] = 1

    return algebraic_connectivity(adj)


# =====================================================================
# Training
# =====================================================================

def compute_returns(rewards, gamma=GAMMA):
    """Discounted returns: R_t = Σ_{t'>=t} γ^(t'-t) r_{t'}."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train(args):
    """Train weight network with REINFORCE + batch baseline."""
    weight_net = WeightNet()
    optimizer = torch.optim.Adam(weight_net.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in weight_net.parameters())

    # Training configs: all (n, m) with baseline data
    train_configs = []
    for n in TRAIN_N:
        max_m = n * (n - 1) // 2
        for m in range(n, max_m + 1):
            if get_best_baseline(n, m) > 0:
                train_configs.append((n, m))

    if not train_configs:
        # Fallback: use all configs even without baseline data
        for n in TRAIN_N:
            max_m = n * (n - 1) // 2
            for m in range(n + 1, max_m + 1):
                train_configs.append((n, m))

    print(f"Training: {args.episodes} episodes, batch={args.batch}, "
          f"lr={args.lr}")
    print(f"Network: {num_params} params | K={K_EIGVECS} eigenvectors")
    print(f"Configs: {len(train_configs)} (n,m) pairs from n={TRAIN_N}")
    print(f"Temperature: {args.tau_start:.3f} → {args.tau_end:.3f}")
    print()

    episode = 0
    reward_history = []
    best_avg = -float('inf')
    best_state = None
    t0 = time.time()

    while episode < args.episodes:
        # Anneal temperature
        frac = min(episode / max(args.episodes - 1, 1), 1.0)
        tau = args.tau_start + (args.tau_end - args.tau_start) * frac

        batch_log_probs = []
        batch_returns = []
        batch_episode_rewards = []

        for _ in range(args.batch):
            n, m = train_configs[np.random.randint(len(train_configs))]
            seed = np.random.randint(0, 1_000_000)

            lam2, log_probs, rewards = collect_episode(
                n, m, weight_net, seed, tau=tau, training=True)

            if not rewards:
                episode += 1
                continue

            returns = compute_returns(rewards)
            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)
            batch_episode_rewards.append(sum(rewards))
            episode += 1

        if not batch_log_probs:
            continue

        # REINFORCE with batch baseline
        returns_t = torch.tensor(batch_returns, dtype=torch.float32)
        advantages = returns_t - returns_t.mean()
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std

        log_probs_t = torch.stack(batch_log_probs)
        loss = -(log_probs_t * advantages.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weight_net.parameters(), 1.0)
        optimizer.step()

        # Track
        avg_ep_reward = np.mean(batch_episode_rewards) if batch_episode_rewards else 0
        reward_history.append(avg_ep_reward)

        if len(reward_history) >= 20:
            recent = np.mean(reward_history[-50:])
            if recent > best_avg:
                best_avg = recent
                best_state = {k: v.clone()
                              for k, v in weight_net.state_dict().items()}

        # Log
        if episode % (args.batch * 20) == 0 or episode >= args.episodes:
            elapsed = time.time() - t0
            recent = np.mean(reward_history[-100:]) if reward_history else 0

            with torch.no_grad():
                # Sample weight vector at a "typical" state
                # (small gap, medium density, mid-progress)
                sample = torch.tensor(
                    [0.05, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 0.4, 0.5],
                    dtype=torch.float32)
                w = weight_net(sample).numpy()

            w_str = ' '.join(f'{x:.2f}' for x in w[:5])
            print(f"  ep {episode:6d} | loss {loss.item():+.4f} | "
                  f"reward {recent:+.4f} | τ={tau:.3f} | "
                  f"w₂..₆=[{w_str}] | {elapsed:.0f}s")

    # Load best
    if best_state is not None:
        weight_net.load_state_dict(best_state)
    print(f"\nTraining complete. Best rolling avg: {best_avg:+.4f}")

    return weight_net


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(weight_net, n_values, num_seeds=5):
    """
    Compare learned weights vs standard FV.

    Both use identical init (same spanning tree seeds).
    Only difference: edge scoring function.
    """
    weight_net.eval()

    print(f"\n{'=' * 90}")
    print(f"EVALUATION | seeds={num_seeds} (best-of)")
    print(f"{'=' * 90}")

    all_results = {}

    for n in n_values:
        max_m = n * (n - 1) // 2
        configs = []
        for m in range(n + 1, max_m + 1):
            if get_best_baseline(n, m) > 0:
                configs.append(m)

        if not configs:
            print(f"\n  N={n}: no baseline data")
            continue

        results = []
        for m in configs:
            fv_db = get_fv_baseline(n, m)
            best_bl = get_best_baseline(n, m)

            # FV from spanning tree (controlled comparison)
            fv_vals = [fv_construct(n, m, seed=s) for s in range(num_seeds)]
            fv_st = max(fv_vals)

            # Learned from same spanning trees
            with torch.no_grad():
                learned_vals = [
                    collect_episode(n, m, weight_net, seed=s, training=False)[0]
                    for s in range(num_seeds)
                ]
            learned = max(learned_vals)

            results.append({
                'n': n, 'm': m,
                'fv_db': fv_db, 'best_bl': best_bl,
                'fv_st': fv_st, 'learned': learned,
            })

        all_results[n] = results

        # --- Aggregates ---
        avg_fv_db = np.mean([r['fv_db'] for r in results])
        avg_best = np.mean([r['best_bl'] for r in results])
        avg_fv_st = np.mean([r['fv_st'] for r in results])
        avg_learned = np.mean([r['learned'] for r in results])

        # Wins (learned vs FV from same init — the fair comparison)
        w_vs_st = sum(1 for r in results
                      if r['learned'] > r['fv_st'] + 1e-6)
        t_vs_st = sum(1 for r in results
                      if abs(r['learned'] - r['fv_st']) < 1e-6)
        l_vs_st = len(results) - w_vs_st - t_vs_st

        w_vs_db = sum(1 for r in results
                      if r['learned'] > r['fv_db'] + 1e-6)
        w_vs_best = sum(1 for r in results
                        if r['learned'] > r['best_bl'] + 1e-6)

        in_train = "TRAIN" if n in TRAIN_N else "TEST"

        print(f"\n  N={n} ({len(results)} configs) [{in_train}]")
        print(f"  {'Method':<22} {'avg λ₂':>8} {'vs FV_db':>9} {'vs best':>9}")
        print(f"  {'-' * 52}")
        print(f"  {'FV (database)':<22} {avg_fv_db:8.3f} {'100.0%':>9} "
              f"{avg_fv_db / avg_best * 100:8.1f}%")
        print(f"  {'FV (spanning tree)':<22} {avg_fv_st:8.3f} "
              f"{avg_fv_st / avg_fv_db * 100:8.1f}% "
              f"{avg_fv_st / avg_best * 100:8.1f}%")
        print(f"  {'Learned weights':<22} {avg_learned:8.3f} "
              f"{avg_learned / avg_fv_db * 100:8.1f}% "
              f"{avg_learned / avg_best * 100:8.1f}%")
        print(f"  Learned vs FV_ST: {w_vs_st}W {t_vs_st}T {l_vs_st}L "
              f"/ {len(results)}")
        print(f"  Learned vs FV_DB: {w_vs_db}W | vs best_BL: {w_vs_best}W")

        # --- Density breakdown ---
        print(f"  By density:")
        bins = [
            ("sparse  <0.3", 0.0, 0.3),
            ("medium 0.3-0.6", 0.3, 0.6),
            ("dense   ≥0.6", 0.6, 1.01),
        ]
        for label, lo, hi in bins:
            sub = [r for r in results if lo <= r['m'] / max_m < hi]
            if not sub:
                continue
            avg_l = np.mean([r['learned'] for r in sub])
            avg_f = np.mean([r['fv_st'] for r in sub])
            wins = sum(1 for r in sub if r['learned'] > r['fv_st'] + 1e-6)
            print(f"    {label}: learned={avg_l:.3f} fv_st={avg_f:.3f} "
                  f"Δ={avg_l - avg_f:+.3f} ({wins}W/{len(sub)})")

    return all_results


# =====================================================================
# Weight analysis
# =====================================================================

def analyze_weights(weight_net):
    """Probe the learned weight function at interpretable states."""
    weight_net.eval()

    print(f"\n{'=' * 90}")
    print(f"WEIGHT ANALYSIS: what the network learned")
    print(f"{'=' * 90}")

    # Header
    evec_labels = [f'w{k}' for k in range(2, 2 + K_EIGVECS)]
    hdr = '  '.join(f'{l:>5}' for l in evec_labels)
    print(f"\n  {'Scenario':<35} {hdr}")
    print(f"  {'-' * (35 + 6 * K_EIGVECS)}")

    scenarios = [
        # (label, gap_ratios[7], density, progress)
        ("FV-friendly: large gap, mid",
         [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.5, 0.5]),
        ("Degenerate λ₂≈λ₃, sparse",
         [0.01, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.15, 0.3]),
        ("Degenerate λ₂≈λ₃, dense",
         [0.01, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.7, 0.5]),
        ("Degenerate λ₂≈λ₃≈λ₄",
         [0.01, 0.02, 0.8, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5]),
        ("Triple degen λ₂≈λ₃≈λ₄≈λ₅",
         [0.01, 0.02, 0.03, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5]),
        ("Well-separated, early step",
         [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 0.3, 0.1]),
        ("Well-separated, late step",
         [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 0.7, 0.9]),
        ("Sparse, early",
         [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 0.1, 0.1]),
        ("Dense, late",
         [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 0.8, 0.9]),
    ]

    with torch.no_grad():
        for label, state_list in scenarios:
            state = torch.tensor(state_list, dtype=torch.float32)
            w = weight_net(state).numpy()
            vals = '  '.join(f'{x:5.3f}' for x in w)
            print(f"  {label:<35} {vals}")

    # FV reference
    print(f"\n  {'FV (reference)':<35} "
          + '  '.join(f'{1.0 if k == 0 else 0.0:5.3f}'
                      for k in range(K_EIGVECS)))

    # Sweep: fix everything, vary gap ratio
    print(f"\n  Gap sensitivity (vary λ₂-λ₃ gap, others fixed):")
    print(f"  {'gap23':>8} {hdr}")
    gaps_to_test = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    with torch.no_grad():
        for g in gaps_to_test:
            state = torch.tensor(
                [g, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 0.4, 0.5],
                dtype=torch.float32)
            w = weight_net(state).numpy()
            vals = '  '.join(f'{x:5.3f}' for x in w)
            print(f"  {g:8.3f} {vals}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Learned Spectral Weights PoC: "
                    "blend eigenvector gaps via REINFORCE")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Training episodes")
    parser.add_argument("--batch", type=int, default=16,
                        help="Episodes per REINFORCE update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau-start", type=float, default=0.1,
                        help="Initial sampling temperature")
    parser.add_argument("--tau-end", type=float, default=0.02,
                        help="Final sampling temperature")
    parser.add_argument("--eval-seeds", type=int, default=5,
                        help="Seeds per config for eval (best-of)")
    parser.add_argument("--eval-n", type=str, default="8,10,12,16,24",
                        help="Comma-separated n values for evaluation")
    args = parser.parse_args()

    # Train
    weight_net = train(args)

    # Evaluate
    eval_n = [int(x) for x in args.eval_n.split(',')]
    evaluate(weight_net, eval_n, num_seeds=args.eval_seeds)

    # Analyze
    analyze_weights(weight_net)


if __name__ == "__main__":
    main()
