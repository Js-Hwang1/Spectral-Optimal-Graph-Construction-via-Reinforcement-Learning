# G(n,m) Algebraic Connectivity Maximization — REFINE v5

## Problem

Given $(n, m)$, construct a simple undirected graph with $n$ nodes and $m$ edges that maximizes $\lambda_2(L)$, the **algebraic connectivity** — the second-smallest eigenvalue of the graph Laplacian $L = D - A$.

**Hard constraints:**
- **Training**: $O(N^3)$ eigensolvers allowed (correct reward signals)
- **Inference**: $O(N^2)$ only — no eigensolvers, no dense $N \times N$ matmul
- **Foundational**: Train on $n = 8\text{–}16$, generalize to $n = 32, 64, 128, 256, 512, 1024$

---

## Algorithm: REFINE(K, Δ)

### Pseudocode — Full Episode

```
REFINE(n, m, K=20, Δ=2):
  G ← RandomSpanningTree(n) + RandomEdges(m - (n-1))
  v₂ ← None                                               # cold Lanczos start

  for k = 0 .. K-1:
    ────── ADD PHASE ──────
    v₂, v₃, λ₂, λ₃ ← Lanczos(G, k=15, warm=v₂)          # O(N²), warm-started
    edge_feat[i,j] ← [n·|v₂ᵢ-v₂ⱼ|², n·|v₃ᵢ-v₃ⱼ|²,      # (N, N, 6)
                       dᵢ/(n-1), dⱼ/(n-1), (λ₃-λ₂)/n, k/K]
    add_logits ← ADD_MLP(edge_feat)                         # (N, N) scores
    targets ← SAMPLE_ADD(add_logits, A, Δ)                  # (N, Δ) targets
    N' ← ApplyAdd(G, targets)
    N'_eff ← CapToSafeBudget(G)

    ────── REMOVE PHASE ──────
    v₂, v₃, λ₂, λ₃ ← Lanczos(G, k=15, warm=v₂)          # O(N²), re-run
    edge_feat_rem ← [same 6 features on updated graph]
    rem_logits ← REM_MLP(edge_feat_rem)                     # (N, N) scores
    bridges ← Tarjan(G)
    edges ← SAMPLE_REMOVE(rem_logits, A, bridges, N'_eff)
    ApplyRemove(G, edges)

    ────── REWARD (training only) ──────
    r_k ← ExactEigensolver(G).λ₂ - λ₂_prev                 # O(N³), Δλ₂
    assert |E(G)| == m

  return G
```

---

## Why This Design (v5 vs v4)

### Why Edge-Level MLP (Not GNN + Inner Product)

**v4 problem**: GNN takes node features `[degree, v₂, step_frac]` and must learn through message passing + inner product that `|v₂ᵢ - v₂ⱼ|²` is the key signal. This is 3 layers of indirection for something perturbation theory tells us directly.

**v5 solution**: Compute `|v₂ᵢ - v₂ⱼ|²` explicitly as an edge feature and feed it to a simple MLP. The model gets the greedy signal for free and learns **when to deviate from greedy** using λ₃-λ₂ and v₃ gap context.

### Why v₃ and Spectral Gap Features

- **Spectral gap (λ₃ - λ₂)**: When small, the Fiedler direction is unstable — greedy Fiedler-gap strategy breaks down here. The MLP learns to weight v₃ information more heavily when the gap is small.
- **v₃ gap `|v₃ᵢ - v₃ⱼ|²`**: Reveals the *next* bottleneck after the current one shifts. Enables non-greedy multi-step planning.

### Why Feature Normalization

All features are O(1) regardless of N for generalization:
- `n · |v₂ᵢ - v₂ⱼ|²` compensates for v₂ having unit norm over N entries
- `degᵢ / (n-1)` always in [0, 1]
- `(λ₃ - λ₂) / n` bounded
- `step / K` always in [0, 1]

---

## Policy Network (~14K params)

```
Edge features (N, N, 6):
  [n·|v₂_gap|², n·|v₃_gap|², deg_i_norm, deg_j_norm, spec_gap, step_frac]
  ↓
┌────────────────────────┐  ┌────────────────────────┐  ┌──────────────────────┐
│ ADD MLP                │  │ REMOVE MLP             │  │ Value MLP            │
│ Linear(6 → 64) + SiLU │  │ Linear(6 → 64) + SiLU │  │ Linear(5 → 64) + SiLU│
│ Linear(64 → 64) + SiLU│  │ Linear(64 → 64) + SiLU│  │ Linear(64→ 64) + SiLU│
│ Linear(64 → 1)        │  │ Linear(64 → 1)        │  │ Linear(64 → 1)       │
│ → (N,N) logits        │  │ → (N,N) logits        │  │ → V(s) scalar        │
└────────────────────────┘  └────────────────────────┘  └──────────────────────┘
                                                        Graph features (5,):
                                                        [mean_deg, λ₂/n, λ₃/n,
                                                         spec_gap, step_frac]
```

O(N²) per MLP call (N² pairs × fixed-width MLP). No message passing.

---

## Reward Structure

**Per-step reward:**
$$r_k = \lambda_2^{(k+1)} - \lambda_2^{(k)}$$

Computed via exact eigensolver ($O(N^3)$, training only). Disconnection penalty: $r = -10$.

---

## PPO Training

Same PPO objective as v4. Joint log probability:
$$\log \pi(a \mid s) = \underbrace{\sum_{i=1}^{N} \sum_{d=1}^{\Delta} \log p_{\text{add}}(t_{i,d} \mid \text{remaining}_d)}_{\text{ADD: N·Δ categoricals}} + \underbrace{\sum_{r=1}^{R} \log p_{\text{rem}}(e_r \mid \text{remaining}_r)}_{\text{REMOVE: sequential conditional}}$$

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| K (Lanczos steps) | 20 | Constant, independent of N |
| Δ (edges per node) | 2 | ADD throughput multiplier |
| γ | 0.99 | Discount factor |
| λ_GAE | 0.95 | GAE bias-variance tradeoff |
| clip ε | 0.2 | PPO clipping |
| PPO epochs | 4 | Updates per episode |
| LR | 3e-4 | AdamW with weight decay 1e-4 |
| Grad clip | 1.0 | Max gradient norm |
| Entropy coef | 0.1 → 0.01 | Linear anneal over training |

---

## Complexity Analysis (Inference)

| Component | Operation | Cost |
|-----------|-----------|------|
| Lanczos (×2) | k=15 mat-vecs, extracts v₂+v₃ | $O(N^2)$ |
| Edge features (×2) | pairwise gaps + degrees | $O(N^2)$ |
| ADD MLP | (N², 6) → MLP → (N², 1) | $O(N^2 \cdot d^2)$ |
| REM MLP | same | $O(N^2 \cdot d^2)$ |
| Bridge detection | Tarjan's algorithm | $O(N + M)$ |
| Sampling | Per-row / sequential conditional | $O(N^2)$ |

**Per-step: $O(N^2)$. Per-episode ($K=20$): $O(N^2)$. No eigensolvers at inference.**

---

## File Structure

```
rl/
├── train_refine.py          # Pure PPO training (REFINE v5)
├── eval_refine.py           # Eval vs FV/ER/SW baselines
├── envs/
│   └── refine_env.py        # REFINE environment (Lanczos ext + edge features)
├── models/
│   └── refine_policy.py     # Edge MLP policy (ADD/REM/Value MLPs)
└── utils/
    ├── spectral.py          # lanczos_fiedler_ext, exact_lambda2, find_bridges
    └── graph6.py            # Graph6 encoding
```

---

## Scaling Story

**What transfers across N:**
- MLP weights are size-agnostic (same MLP applied per-edge on any graph)
- All 6 edge features are normalized to O(1) regardless of N
- Δ is constant — each node adds Δ edges, total adds scale as N·Δ naturally
- Lanczos extracts v₂, v₃ from single run regardless of N

**What doesn't transfer (and why it's fine):**
- Exact eigensolvers ($O(N^3)$) — training only, replaced by Lanczos at inference
- MLP applied to N² pairs at inference — O(N²) total, acceptable

---

## Experimental Results

### v4 Results (GNN, 50K HPC) — Baseline

| Metric | n=16 | n=24 | n=36 | Total |
|--------|------|------|------|-------|
| Wins (beat ALL baselines) | 0/106 | 1/254 | 22/596 | 23/956 |
| Avg RL / best ratio | 60.8% | 62.2% | 59.8% | ~60% |

**v4 verdict**: GNN couldn't learn effective structural patterns. ~60% of baseline quality.

### v5 Architecture Change

Replaced GNN+inner product (166K params) with edge-level MLP on spectral features (14K params). Key insight: give the model `|v₂ᵢ-v₂ⱼ|²` directly instead of making it rediscover this through message passing.

**200-episode local smoke test (n=8..16):**

| Window | avg Δλ₂ | Disconnections |
|--------|---------|----------------|
| First 50 | -0.113 | 6/50 (12%) |
| Last 50 | -0.005 | 3/50 (6%) |

Learning signal significantly faster than v4 (which was at -0.25 after first 50 episodes).
