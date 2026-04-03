# REFINE Architecture — Implementation Summary

## Overview

REFINE is a two-phase iterative graph refinement architecture for maximizing algebraic connectivity (lambda2) of G(n,m) graphs. Each step decouples the action into:
1. **ADD phase** (node-centric): each node picks a non-neighbor to connect to (N independent categoricals)
2. **REMOVE phase** (edge-level): sample exactly N' edges from a categorical over all valid edges (without replacement)

**Strict (N,M) compliance**: the graph MUST return to exactly M edges after every step. The environment asserts this.

Training is a two-stage pipeline:
- **Phase 1**: Behavioral Cloning (BC) from a Fiedler vector teacher
- **Phase 2**: PPO fine-tuning from Phase 1 checkpoint

---

## Architecture Details

### Policy Network: `models/refine_policy.py` (~165,761 params)

```
Input: node_feat (N, 3), adj (N, N)
  -> input_proj: Linear(3 -> 128)
  -> 3x SAGEConvLayer(128, 128):
       h' = W_self(h) + W_neigh(D^{-1}A @ h)
       out = LayerNorm(SiLU(h') + h)   [residual + LayerNorm]
  -> H_shared (N, 128)
  -> ADD Head:  Linear(128->128) + SiLU + Linear(128->64) -> H_add (N, 64)
                H_add @ H_add^T / sqrt(64) -> (N, N) logits
  -> REM Head:  Linear(128->128) + SiLU + Linear(128->64) -> H_rem (N, 64)
                H_rem @ H_rem^T / sqrt(64) -> (N, N) logits
  -> Value Head: mean_pool(H_shared) -> Linear(128->128) + SiLU + Linear(128->1)
```

**Key design choices:**
- **SAGEConv with mean aggregation** (D^{-1}A normalization): size-agnostic, generalizes from n=16 to n=1024 without scale issues (unlike GIN's sum aggregation)
- **Inner product scoring** (H @ H^T): O(N^2 x d_head) for fixed d_head=64. Symmetric (correct for undirected graphs). Scaled by 1/sqrt(d_head)
- **Weight init**: orthogonal with gain=0.5
- **No eigensolvers at inference**: entire forward pass is O(N^2)

**ADD head**: per-row distribution (N independent categoricals)
- ADD row i: softmax over non-neighbors of i
- Mask: `(adj == 0) & ~eye`

**REM head**: produces per-node (N,N) logits, but these are converted to **edge-level scores** downstream:
- `edge_score(i,j) = rem_logits[i,j] + rem_logits[j,i]` for canonical edge (i < j)
- Valid edges: existing, non-bridge, upper triangle
- Mask: `(adj > 0) & ~eye & ~bridge_mask`

### Environment: `envs/refine_env.py`

**State:** `adj (N, N)`, `degrees (N,)`, `n`, `m`, `m_current`, `step`, `lambda2`

**Episode flow (strict M compliance):**
```
1. reset(n, m):
     Build random spanning tree + random edges via build_random_tree_initial()
     Compute initial lambda2 via exact eigensolver (training only)

2. For k = 0..K-1 (K=20):
   a. ADD phase — add_phase(targets: (N,)):
        Each node i proposes edge (i, targets[i])
        Symmetry: canonical edge = (min(i,j), max(i,j))
        If i->j and j->i both proposed, only one undirected edge added
        Stores _added_edges list for potential rollback
        Returns n_added

   b. cap_to_safe_budget() -> effective_n:
        Counts non-bridge edges via Tarjan's algorithm
        If safe_removable < n_added: rolls back last (n_added - safe_removable) added edges
        Returns effective_n = min(n_added, safe_removable)

   c. REMOVE phase — remove_phase(edge_indices: (K, 2)):
        Removes exactly the K specified edges (K == effective_n)
        Guarantees n_removed == effective_n == n_added (after cap)

   d. compute_reward() -> delta_lambda2:
        ASSERTS m_current == m (strict compliance)
        new_lambda2 = exact eigensolver (O(N^3), training only)
        reward = new_lambda2 - old_lambda2
        Penalty of -10.0 if graph becomes disconnected (lambda2 < 1e-6)

3. Episode ends after K steps
```

**Node features** (3-dim, O(N^2)):
- degree_norm: degree / (n-1)
- 2-hop reachability: neighbor degree sum proxy (NOT adj @ adj)
- 3-hop reachability: similar proxy

**Bridge detection:** Tarjan's algorithm O(N+M), computed in cap_to_safe_budget() and get_bridge_mask().

### Complexity (Inference — strictly O(N^2))

| Component | Per-call | Calls/step | Total/step |
|-----------|----------|------------|------------|
| Node features (3-dim) | O(N^2) | 2 | O(N^2) |
| SAGEConv encode (3 layers) | O(N^2 d) | 2 | O(N^2) |
| H @ H^T scoring | O(N^2 d_head) | 2 | O(N^2) |
| ADD masking + argmax | O(N^2) | 1 | O(N^2) |
| Bridge detection (Tarjan) | O(N+M) | 1 | O(N^2) |
| cap_to_safe_budget | O(N^2) | 1 | O(N^2) |
| Edge scores + top-K | O(M) | 1 | O(N^2) |

**Total per step: O(N^2). Total per episode (K=20): O(N^2). No eigensolvers.**

**Forbidden ops**: No eigensolvers, no N×N @ N×N matmul. `H @ H^T` is `(N, d_head) @ (d_head, N)` = O(N^2 × d_head).

---

## Training Pipeline

### Phase 1: Behavioral Cloning — `train_refine_phase1.py`

**Teacher signal (O(N^3), training only):**
1. Compute Fiedler vector v2 via `np.linalg.eigh(L)` where L = D - A
2. ADD teacher (per-row): `P_add[i,j] = (v2[i] - v2[j])^2` over non-neighbors of i, row-normalized
3. REMOVE teacher (edge-level): `score(i,j) = 1 / ((v2[i] - v2[j])^2 + eps)` over valid edges, softmax-normalized

**Intuition:** ADD teacher connects nodes far apart in the Fiedler embedding (to increase algebraic connectivity). REMOVE teacher removes edges between nodes close in the Fiedler embedding (they are already well-connected, so the edge is redundant).

**Training loop:**
```
For each epoch (100 epochs x 500 episodes/epoch):
  For each episode:
    Sample random (n, m) from [8, 16]
    Build random graph
    For k = 0..K-1:
      1. Compute Fiedler vector v2 (O(N^3) teacher)
      2. Forward policy -> model ADD logits (N, N)
      3. ADD loss: per-row KL(teacher || model), averaged over valid rows
      4. Execute ADD with TEACHER targets (teacher drives trajectory)
      5. cap_to_safe_budget() -> effective_n
      6. Recompute Fiedler vector on post-ADD graph
      7. Compute bridge mask + valid edges
      8. Forward policy -> model REMOVE logits -> edge scores
      9. REMOVE loss: edge-level KL(teacher || model), single categorical over valid edges
     10. Sample effective_n from teacher without replacement -> edge_indices
     11. Execute remove_phase(edge_indices)
     12. Assert m_current == m
    Backward with gradient accumulation (accum_steps=4)
```

**Loss:**
- ADD: `F.kl_div(log_softmax(model_logits[valid_rows]), teacher[valid_rows], reduction='batchmean')` — per-row KL average
- REMOVE: `F.kl_div(log_softmax(model_edge_scores), teacher_edge_probs, reduction='sum')` — edge-level KL of single categorical

**Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4, CosineAnnealingLR (eta_min=lr*0.01), grad_clip=1.0.

**Key detail:** During Phase 1, the **teacher drives the trajectory** (not the model). This keeps the training distribution stable — standard behavioral cloning practice.

### Phase 2: PPO Fine-Tuning — `train_refine_phase2.py`

Loads Phase 1 `best.pt` checkpoint and fine-tunes with PPO.

**Episode collection:**
```
For k = 0..K-1:
  1. State S_t = graph BEFORE ADD phase (N, M edges)
  2. Forward ADD head -> sample N targets (one per node, from categorical)
  3. Execute ADD phase -> n_added edges
  4. cap_to_safe_budget() -> effective_n
  5. Recompute features + bridge mask on post-ADD graph
  6. Forward REMOVE head -> compute edge scores -> sample effective_n edges without replacement
  7. Execute remove_phase(edge_indices) -> graph back to (N, M)
  8. Compute reward via exact eigensolver: delta_lambda2
  Store: (features_add, adj_add, add_targets, features_rem, adj_rem, bridge_mask,
          edge_indices, effective_n, log_prob, value, reward)
```

**Value function (MDP compliant):**
- `value = V(S_t)` computed from pre-ADD state only
- The step is the entire ADD+REMOVE cycle; S_t is the (N,M) graph before ADD
- Do NOT compute or use value from the transient (N, M+N') state

**Joint log-probability:**
```
add_log_prob = sum_i log P_add(target_i | S_t)    — N independent categoricals
rem_log_prob = sum_k log P_rem(e_k | remaining)    — sequential conditional (without-replacement)
total_log_prob = add_log_prob + rem_log_prob
```

**REMOVE without-replacement sampling:**
- Edge scores: `score(i,j) = rem_logits[i,j] + rem_logits[j,i]`
- Sequential conditional: for each of effective_n draws, softmax over remaining valid edges, sample one, mask it out
- Log-prob: sum of log P(e_k | remaining edges) for k = 1..effective_n

**PPO update (per episode):**
- GAE advantages: gamma=0.99, lambda=0.95
- Advantage normalization: (adv - mean) / (std + 1e-8)
- Clipped surrogate: epsilon=0.2
- Value loss: MSE, coefficient=0.5
- Entropy bonus: scheduled 0.1 -> 0.01 linearly over training
- 4 PPO epochs per episode
- Gradient clipping: 0.5
- Optimizer: Adam, lr=3e-4, CosineAnnealingLR (eta_min=lr*0.1)

**PPO re-evaluation:** `evaluate_remove_log_prob()` recomputes sequential conditional log-prob under the updated policy. Uses `remaining.clone()` to avoid in-place autograd issues. Gradient flows through edge_scores -> rem_logits -> policy weights.

### Evaluation: `eval_refine.py`

- Load checkpoint, run K=20 REFINE steps with **deterministic actions**
- ADD: per-row argmax over masked logits
- REMOVE: edge-level top-N' selection (`torch.topk(edge_scores, effective_n)`) — O(M) = O(N^2)
- cap_to_safe_budget() ensures M compliance
- Compare final lambda2 against FV, ER, SW baselines from HuggingFace
- Multiprocessing: each (n, m) config evaluated in parallel
- CSV export of results

---

## Files

| File | Lines | Description |
|------|-------|-------------|
| `models/refine_policy.py` | 260 | RefinePolicy: SAGEConv GNN + inner-product ADD/REM heads + value head |
| `envs/refine_env.py` | 290 | RefineEnv: strict M compliance, cap_to_safe_budget(), remove_phase(edge_indices) |
| `train_refine_phase1.py` | 415 | Phase 1 BC: per-row ADD KL + edge-level REMOVE KL |
| `train_refine_phase2.py` | 390 | Phase 2 PPO: edge-level REMOVE sampling, V(S_t) value, sequential conditional |
| `eval_refine.py` | 280 | Evaluation: deterministic top-K REMOVE, M compliance |
| `slurm/train_refine.slurm` | 80 | HPC job: Phase 1 then Phase 2 sequential |
| `slurm/eval_refine.slurm` | 53 | HPC job: evaluate both checkpoints |

---

## Training Results

### Run 3 (v3) — Strict M Compliance + Edge-Level REMOVE + MDP Value

Trained on nvwulf (job 15561), 62 CPUs, ~65 min total.

#### Phase 1 — Behavioral Cloning (32.1 min)

| Metric | Epoch 1 | Epoch 25 | Epoch 50 | Epoch 96 (Best) | Epoch 100 |
|--------|---------|----------|----------|-----------------|-----------|
| ADD KL | 0.240 | 0.188 | 0.169 | 0.147 | 0.157 |
| REM KL | 2.464 | 2.165 | 1.978 | 1.751 | 1.925 |
| Total | 2.705 | 2.353 | 2.147 | **1.897** | 2.082 |

- Best checkpoint saved at epoch 96 (total_loss=1.897)
- REMOVE KL is higher than Run 2 (1.90 vs 0.55) because it's now a single edge-level categorical over ~M edges rather than N per-row categoricals — much harder distribution to learn
- ADD KL is comparable (0.147 vs 0.137)

#### Phase 2 — PPO Fine-Tuning (32.7 min)

| Window | Avg lambda2 | Avg Improvement | Best Avg Imp |
|--------|-------------|-----------------|--------------|
| Ep 0-1K | 3.883 | +0.163 | 0.263 |
| Ep 1K-2K | 4.116 | +0.193 | 0.305 |
| Ep 2K-3K | 3.752 | +0.169 | 0.305 |
| Ep 3K-4K | 4.167 | +0.182 | 0.305 |
| Ep 4K-5K | 4.027 | +0.224 | 0.342 |
| Ep 5K-6K | 4.200 | +0.252 | 0.421 |
| Ep 6K-7K | 3.907 | +0.112 | 0.421 |
| Ep 7K-8K | 3.914 | +0.266 | 0.421 |
| Ep 8K-9K | 3.992 | +0.291 | 0.500 |
| Ep 9K-10K | 4.013 | +0.260 | 0.500 |

- **PPO now consistently improves** — avg improvement is **positive** across all 10K episodes
- Best avg improvement: **+0.500** (vs -1.394 in Run 2)
- No disconnected graph collapses (strict M compliance prevents edge drift)

---

## Evaluation Results (Run 3 — v3)

Evaluated with deterministic inference, 5 trials per (n,m), n=8,10,12,16,24,32,36 (1537 total configs).

### Phase 1 BC Checkpoint (best.pt, epoch 96)

| n | Configs | Wins | Win Rate | Avg RL lambda2 | Avg Best Baseline | RL/Best |
|---|---------|------|----------|----------------|-------------------|---------|
| 8 | 22 | 5 | 22.7% | 2.834 | 3.515 | 80.6% |
| 10 | 37 | 5 | 13.5% | 3.417 | 4.333 | 78.9% |
| 12 | 56 | 4 | 7.1% | 4.075 | 5.153 | 79.1% |
| 16 | 106 | 4 | 3.8% | 5.265 | 6.886 | 76.5% |
| 24 | 254 | 8 | 3.1% | 7.562 | 10.466 | 72.2% |
| 32 | 466 | 10 | 2.1% | 8.399 | 14.101 | 59.6% |
| 36 | 596 | 16 | 2.7% | 9.152 | 15.933 | 57.4% |
| **Total** | **1537** | **52** | **3.4%** | — | — | — |

- >= 99% of best baseline: 62/1537 (4.0%)
- >= 95% of best baseline: 79/1537 (5.1%)
- **Generalizes to unseen sizes**: wins at n=24 (8), n=32 (10), n=36 (16)
- RL/Best ratio: 80.6% at n=8 -> 57.4% at n=36 (gradual degradation, not collapse)

### Phase 2 PPO Checkpoint (best.pt, episode 8900)

| n | Configs | Wins | Win Rate | Avg RL lambda2 | Avg Best Baseline | RL/Best |
|---|---------|------|----------|----------------|-------------------|---------|
| 8 | 22 | 0 | 0% | 2.852 | 3.515 | 81.1% |
| 10 | 37 | 1 | 2.7% | 3.590 | 4.333 | 82.9% |
| 12 | 56 | 3 | 5.4% | 4.266 | 5.153 | 82.8% |
| 16 | 106 | 2 | 1.9% | 5.351 | 6.886 | 77.7% |
| 24 | 254 | 0 | 0% | 7.207 | 10.466 | 68.9% |
| 32 | 466 | 0 | 0% | 7.423 | 14.101 | 52.6% |
| 36 | 596 | 0 | 0% | 7.904 | 15.933 | 49.6% |
| **Total** | **1537** | **6** | **0.4%** | — | — | — |

- >= 99% of best baseline: 18/1537 (1.2%)
- >= 95% of best baseline: 29/1537 (1.9%)
- PPO slightly improves avg lambda2 at n=8-16 vs Phase 1, but loses generalization at n=24-36
- Fewer wins than Phase 1 at all sizes

---

## Comparison Across Runs

### Phase 1 BC (training-range evaluation, n=8-16)

| Metric | Run 1 | Run 2 | Run 3 (v3) |
|--------|-------|-------|------------|
| Model params | 42K | 165K | 165K |
| REMOVE action space | Global upper-tri | Node-centric per-row | Edge-level categorical |
| M compliance | No (drift) | No (drift) | **Strict (assertion)** |
| Value function | — | (add+rem)/2 | V(S_t) only |
| Best KL loss | 1.398 | 0.547 | 1.897* |
| ADD KL | 0.151 | 0.137 | 0.147 |
| REM KL | 1.247 | 0.410 | 1.751* |
| Avg RL lambda2 (n=8) | — | 1.092 | 2.834 |
| Avg RL lambda2 (n=16) | — | 3.101 | 5.265 |

*Run 3 REM KL is not comparable — it's edge-level KL (single categorical over ~M edges) vs per-row KL (N categoricals over ~N entries each). The distributions have fundamentally different cardinalities.

### Phase 2 PPO

| Metric | Run 2 | Run 3 (v3) |
|--------|-------|------------|
| Avg improvement | **-1.394** | **+0.260** |
| PPO direction | Degrades BC | **Improves BC** |
| Disconnected graphs | 85/10K (0.85%) | 0 |
| Best avg improvement | -0.983 | +0.500 |

The strict M compliance and MDP-compliant value function fixed PPO. It now consistently improves lambda2 instead of degrading the policy.

### Generalization (new in Run 3)

| n | Phase 1 Wins | Phase 1 RL/Best | Phase 2 Wins | Phase 2 RL/Best |
|---|-------------|-----------------|-------------|-----------------|
| 24 | 8/254 (3.1%) | 72.2% | 0/254 | 68.9% |
| 32 | 10/466 (2.1%) | 59.6% | 0/466 | 52.6% |
| 36 | 16/596 (2.7%) | 57.4% | 0/596 | 49.6% |

Phase 1 BC generalizes better than Phase 2 PPO. At n=36, BC still achieves 57% of baseline lambda2 and wins 16 configs, while PPO drops to 50% with 0 wins.

---

## Key Findings

1. **Strict M compliance works**: The cap_to_safe_budget() + assertion approach eliminates edge count drift entirely. Zero disconnected graphs in PPO training.

2. **PPO is now directionally correct**: With MDP-compliant value V(S_t) and strict M preservation, PPO consistently improves lambda2 (avg +0.26/episode vs -1.39 before). The fix from (add_val+rem_val)/2 to V(S_t) alone was critical.

3. **Edge-level REMOVE is harder to learn**: The REMOVE KL in Phase 1 plateaus at ~1.75 (vs ~0.41 for per-row). This is expected — a single categorical over ~M edges is a much larger distribution than N categoricals over ~N entries each. But the edge-level formulation is mathematically correct for the without-replacement sampling.

4. **BC generalizes better than PPO**: Phase 1 checkpoint wins 52/1537 configs across all sizes, while Phase 2 only wins 6. PPO improves average lambda2 at training sizes but hurts the argmax (deterministic) policy at generalization sizes.

5. **Avg RL lambda2 significantly improved**: At n=8: 1.09 -> 2.83 (+160%). At n=16: 3.10 -> 5.27 (+70%). The strict M compliance prevents the graph from degrading.

---

## Remaining Challenges

1. **PPO hurts generalization**: Phase 2 improves stochastic policy (positive avg improvement during training) but the deterministic argmax policy degenerates at larger n. This is the classic BC-vs-RL tradeoff — RL overfits to the training distribution.

2. **RL/Best ratio degrades with n**: 80.6% at n=8 -> 57.4% at n=36. Need better generalization — potentially curriculum learning, larger training range, or architecture changes.

3. **Edge-level REMOVE KL convergence**: The edge-level distribution is harder to learn. May benefit from longer Phase 1 training, different loss weighting, or auxiliary objectives.

4. **Win rate is low for dense graphs**: Most wins are in sparse configs. Dense graphs remain challenging.
