# REFINE v9f — Aligned PPO + Fiedler-Only Features

## Two Key Fixes in v9f

### Fix 1: PPO Alignment (v9)

In v7-rescore, there was a fundamental **PPO mismatch**:

1. At collection time, edges are chosen **sequentially**: after adding edge 1, RR updates the spectral state, logits are re-scored, and edge 2 is sampled from **updated** logits.
2. At PPO re-evaluation time, policy gradients are computed against the **initial** logits (before any edges were added in that phase).
3. PPO is optimizing a policy that **never actually generated the actions**.

**Fix**: Each edge op rebuilds full features and stores them. PPO re-evaluates each action using the exact state the agent saw → correct gradients.

### Fix 2: Feature Ablation (v9f)

A PoC (`PoC/feature_ablation_poc.py`) revealed that the 10-dim features were mostly noise:

| Feature | Correlation with Δλ₂ | Verdict |
|---------|----------------------|---------|
| v2 gap (Fiedler) | **r = 0.904** | **KEEP** — overwhelmingly dominant signal |
| deg_i | r = 0.171 | **KEEP** — structural info |
| deg_j | r = 0.089 | **KEEP** — structural info |
| step/K | r = 0.000 | **KEEP** — budget awareness |
| v3 gap | r = -0.133 | REMOVE — near-zero, adds noise |
| v4 gap | r = -0.148 | REMOVE — near-zero, adds noise |
| v5 gap | r = -0.086 | REMOVE — near-zero, adds noise |
| gap λ₃-λ₂ | r = 0.381 | REMOVE — MLP test showed it hurts top-K accuracy |
| gap λ₄-λ₃ | r = -0.038 | REMOVE — near-zero |
| gap λ₅-λ₄ | r = 0.138 | REMOVE — near-zero |

Key PoC findings:
- **fiedler_only (4-dim) beat full (10-dim) on top-5% edge selection**: 75% vs 66.7%
- **fiedler_global (7-dim, adding spectral gaps) was worst**: 41.7% top-5%
- The extra features gave the MLP 6 dimensions of noise to overfit to, hurting generalization especially at mid-density

**Fix**: Strip to 4-dim edge features (Fiedler gap, deg_i, deg_j, step) and 3-dim graph features (mean_degree, λ₂/n, step).

## Evolution: v7 → v7-rescore → v8 → v9 → v9f

| Version | Reward | PPO | Features | Best Δλ₂ | Stability |
|---------|--------|-----|----------|----------|-----------|
| v7-batch | Batch (shared) | Aligned (trivial) | 10-dim | 0.5588 | Collapsed @1.2k |
| v7-rescore | Batch + isolated REM | **Misaligned** | 10-dim | 0.7112 | Crashed @25k (NaN) |
| v8 per-edge | Per-edge | Aligned | 10-dim | 0.8937 | Stable 100k |
| v9 | Batch + isolated REM | Aligned | 10-dim | 0.8888 | Stable |
| **v9f** | **Batch + isolated REM** | **Aligned** | **4-dim** | **training** | **Stable** |

## Eval Results

### v9f-fiedler (4-dim, only 23% trained!) vs v9-full (10-dim, 46% trained)

| n | v9-full (10-dim) | v9f-fiedler (4-dim) | Δ ratio |
|---|-----------------|---------------------|---------|
| 16 | 2W / 7T, 88.5% | 0W / **16T**, 88.9% | **+0.4%** |
| 24 | 1W / 10T, 85.1% | 0W / **42T**, 87.0% | **+1.9%** |
| 36 | 0W / 23T, 84.5% | 2W / **103T**, 87.3% | **+2.8%** |

The improvement **grows with n** — the noise features were hurting generalization most. At n=36, ties jumped from 23 → 103 (4.5×) and v9f scored its first wins, all at half the training.

### Full version comparison at n=16

| Version | Wins | Ties | Ratio | Training |
|---------|------|------|-------|----------|
| v7-batch | 0 | — | 86.4% | 50k |
| v7-rescore | 0 | 6 | 88.5% | 50k |
| v8 per-edge | 3 | 5 | 82.3% | 100k |
| v9-full | 2 | 7 | 88.5% | 18.8k (46%) |
| **v9f-fiedler** | **0** | **16** | **88.9%** | **11.7k (23%)** |

### Full version comparison at n=36 (generalization, trained on n≤16)

| Version | Wins | Ties | Ratio |
|---------|------|------|-------|
| v7-rescore | 0 | 17 | 83.4% |
| v9-full | 0 | 23 | 84.5% |
| **v9f-fiedler** | **2** | **103** | **87.3%** |

## Architecture

### Policy Network (~13.6K params)

```
ADD MLP:   (4) → [64] → SiLU → [64] → SiLU → (1)    — independent weights
REM MLP:   (4) → [64] → SiLU → [64] → SiLU → (1)    — independent weights
Value MLP: (3) → [64] → SiLU → [64] → SiLU → (1)    — shared for both phases
```

### Edge Features (4-dim)

```
0: n * |v₂ᵢ - v₂ⱼ|²   — Fiedler gap (r=0.90 with Δλ₂)
1: degᵢ / (n-1)        — source degree
2: degⱼ / (n-1)        — target degree
3: step / K             — budget awareness
```

### Graph Features (3-dim, for value function)

```
0: mean_degree / (n-1)
1: λ₂ / n
2: step / K
```

## Episode Flow (v9f Training)

```
Input: (n, m)
    ↓
Initialize: Ring + random edges → suboptimal graph
    ↓
For k = 0..K-1:                                              (K = 20)
  Lanczos → RR init: V=[v₂..v₉], lams=[λ₂..λ₉]             [O(N²)]
  λ₂_pre = exact eigensolve                                   [O(N³)]

  ─── PHASE 1: ADD (per-edge features, batch reward) ───
  For each of num_swaps adds:
    Rebuild full N×N×4 features from current RR state         [O(N²)]  ← PPO-aligned!
    ADD MLP → score all non-edges                              [O(N²)]
    Sample 1 non-edge (softmax → multinomial)
    Execute add + RR update                                    [O(Nk²)]
    Store transition: (features, mask, action, log_prob, value)
    reward = 0                                                 ← sparse!

  λ₂_post_add = exact eigensolve                              [O(N³)]
  Last ADD transition gets reward = λ₂_post_add - λ₂_pre      ← batch reward!

  ─── PHASE 2: REM (per-edge features, batch reward) ───
  Re-init RR (fresh Lanczos on post-add graph)                 [O(N²)]
  For each of num_swaps removes:
    Rebuild full N×N×4 features from current RR state         [O(N²)]  ← PPO-aligned!
    REM MLP → score all edges (excl bridges, just-added)       [O(N²)]
    Sample 1 edge (softmax → multinomial)
    Execute remove + RR update                                 [O(Nk²)]
    Store transition: (features, mask, action, log_prob, value)
    reward = 0                                                 ← sparse!

  λ₂_post_rem = exact eigensolve                              [O(N³)]
  Last REM transition gets reward = λ₂_post_rem - λ₂_post_add ← batch reward!

GAE propagates rewards backward through each stream:
  ADD stream: [0, 0, ..., 0, r_add] → advantages via γλ discounting
  REM stream: [0, 0, ..., 0, r_rem] → advantages via γλ discounting
```

### Per-Transition Storage

```python
{
    'feat':       (N,N,4)   # Full feature tensor at this step
    'mask':       (N,N)     # Valid action mask at this step
    'flat_idx':   int       # The single action taken (flattened i*n+j)
    'graph_feat': (3,)      # Graph-level features for value function
    'log_prob':   float     # log π(action|state) at collection time
    'value':      float     # V(state) estimate at collection time
    'reward':     float     # 0 for all except last in phase
}
```

## Episode Flow (Inference)

```
For k = 0..K-1:
  Lanczos → RR init                                           [O(N²)]
  Build full N×N×4 features                                    [O(N²)]

  ─── ADD PHASE (batch) ───
  ADD MLP → score all non-edges                                [O(N²)]
  For each of num_swaps adds:
    Pick argmax from current logits
    Execute add + RR update
    Re-score affected rows/cols (2 nodes × O(N))               [O(N)]

  ─── REM PHASE (batch) ───
  Re-init RR, rebuild features                                 [O(N²)]
  REM MLP → score all edges                                    [O(N²)]
  For each of num_swaps removes:
    Pick argmax from current logits (excl bridges)
    Execute remove + RR update
    Re-score affected rows/cols                                [O(N)]

No eigensolves at inference. Total: O(N³) for dense graphs.
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| K (Lanczos snapshots) | 20 |
| swap_frac | 0.05 (5% of edges per step) |
| batch_size | 8 episodes per PPO update |
| PPO epochs | 4 |
| clip_eps | 0.2 |
| GAE γ | 0.99 |
| GAE λ | 0.95 |
| LR | 3e-4 (cosine → 3e-5) |
| Entropy | 0.1 → 0.01 |
| Training range | n=8..16 |

## Key Files

| File | Description |
|------|------------|
| `train_refine_batch.py` | v9f aligned PPO training (4-dim features) |
| `eval_refine.py` | Inference with batch scoring + row/col re-scoring |
| `envs/refine_env.py` | RR state, `init_rr()`, `add_single_edge()`, bridges |
| `models/refine_policy.py` | Independent ADD/REM MLPs (4-dim), value MLP (3-dim) |
| `utils/spectral.py` | `lanczos_fiedler_ext_k()`, `rr_update()` |
| `plot.py` | Plot eval CSVs vs DB baselines |
| `PoC/feature_ablation_poc.py` | Feature correlation & MLP ablation study |

## Next Steps

1. **Complete 50k training** of v9f-fiedler (running on macmini, ~33 min)
2. **Eval at n=16, 24, 36** with full training checkpoint
3. **Scale** to n=64, 128, 256 if generalization holds
4. **Consider** increasing K for large-n inference (PoC confirmed K=100 works at n=1024)
