# REFINE+ (REFINE_P) — O(N³) RL with Rich Spectral Features

## Core Thesis

FV is greedy in v₂-space. ER is greedy in L⁺-space. Both are one-shot constructive heuristics
that can't undo mistakes. REFINE+ uses RL to learn **non-greedy multi-step strategies** over a
**richer feature space** that subsumes both FV's and ER's information — plus structural signals
neither can access.

The O(N³) budget goes to **feature richness**, not just eigensolver accuracy (warm Lanczos is
already 99%). The policy sees everything FV sees, everything ER sees, AND more — then learns
when to sacrifice short-term λ₂ for long-term structural improvement.

---

## Feature Design (10-dim edge features)

### What FV sees: Fiedler gap
```
feat[0] = n * |v₂ᵢ - v₂ⱼ|²
```
FV greedily adds the edge with max feat[0]. This is necessary but not sufficient — it's
myopic and ignores secondary bottlenecks.

### What ER sees: Effective resistance
```
feat[2] = R(i,j) / R_mean    where R(i,j) = L⁺ᵢᵢ + L⁺ⱼⱼ - 2L⁺ᵢⱼ
```
Effective resistance = weighted sum of ALL eigenvector gaps: R(i,j) = Σ_k (v_k[i]-v_k[j])²/λ_k.
This is a global connectivity measure. ER greedily adds max-R edges. Our policy sees this AND
can learn when R disagrees with v₂ gap (which happens at eigenvalue crossings).

### What NEITHER sees:

**Common neighbors** — structural redundancy signal:
```
feat[3] = cn(i,j) / (n-2)    where cn(i,j) = (A²)[i,j]
```
If edge (i,j) has many common neighbors, it's structurally redundant — removing it won't
disconnect anything. If a non-edge (i,j) has zero common neighbors, adding it creates a
new path (high impact). FV/ER don't use this.

**v₃ gap** — secondary bottleneck:
```
feat[1] = n * |v₃ᵢ - v₃ⱼ|²
```
v₃ encodes the second partition direction. A graph with high λ₂ but low λ₃ is fragile —
one perturbation flips which direction is the bottleneck. The policy can learn to balance
both.

**Spectral gaps** — eigenvalue stability:
```
feat[6] = (λ₃ - λ₂) / n     — primary gap (how stable is the Fiedler direction?)
feat[7] = (λ₄ - λ₂) / n     — spectral width (how spread are the bottom eigenvalues?)
```
When λ₃ ≈ λ₂, the Fiedler vector is unstable — small changes flip the bottleneck direction.
The policy can learn to widen this gap for robust connectivity.

### Full 10-dim feature vector per edge (i,j):
```
 0: n * |v₂ᵢ - v₂ⱼ|²           — Fiedler gap (what FV uses)
 1: n * |v₃ᵢ - v₃ⱼ|²           — secondary bottleneck gap
 2: R(i,j) / R_mean             — effective resistance (what ER uses)
 3: cn(i,j) / (n-2)             — common neighbors fraction
 4: degᵢ / (n-1)                — source degree
 5: degⱼ / (n-1)                — target degree
 6: (λ₃ - λ₂) / n              — spectral gap stability
 7: (λ₄ - λ₂) / n              — spectral width
 8: step / K                    — budget awareness
 9: s / num_swaps               — swap progress within step
```

### Graph features (7-dim):
```
 0: mean_degree / (n-1)
 1: λ₂ / n
 2: λ₃ / n
 3: λ₄ / n
 4: (λ₃ - λ₂) / n
 5: R_mean * n                  — mean effective resistance (global connectivity)
 6: step / K
```

---

## O(N³) Complexity Budget

| Component | Cost | When |
|-----------|------|------|
| Eigendecomposition (all eigenvalues/vectors) | O(N³) | Once per K-step |
| L⁺ from eigendecomposition | O(N²) | Once per K-step (just V diag(1/λ) Vᵀ diagonal) |
| R matrix (N×N effective resistance) | O(N²) | Once per K-step (from L⁺ diagonal) |
| A² (common neighbors matrix) | O(N²M) ≤ O(N³) | Once per K-step |
| MLP scoring all N² pairs | O(N²) | Once per K-step |
| Per-swap RR update | O(Nk²) | M/10 times per step |
| Per-swap common neighbors update | O(N) | M/10 times per step |
| Per-swap row-update (logits) | O(N) | M/10 times per step |
| **Total per K-step** | **O(N³)** | |
| **Total per episode (K=20)** | **O(N³)** | |

---

## Feature Update Strategy Between Swaps

After each swap (add edge u,v then remove edge p,q):

| Feature | Update method | Cost |
|---------|--------------|------|
| v₂, v₃ gaps (feat 0,1) | RR subspace rotation | O(Nk²) |
| Effective resistance (feat 2) | **Stale** until next K-step | O(0) |
| Common neighbors (feat 3) | Incremental: cn[u,:] += A[v,:], etc. | O(N) |
| Degrees (feat 4,5) | Direct ±1 update | O(1) |
| Spectral gaps (feat 6,7) | From RR eigenvalues | O(1) |
| Step/swap progress (feat 8,9) | Counter update | O(1) |

Effective resistance is the only stale feature. It changes slowly (rank-1 update per swap)
and is refreshed every K-step. Acceptable for M/10 swaps between refreshes.

---

## Why This Beats FV/ER

1. **Information advantage**: Policy sees FV's signal AND ER's signal AND structural features.
   It can learn to use whichever is most informative for the current graph state.

2. **Non-greedy strategy**: FV always picks max v₂-gap. But sometimes the second-best v₂-gap
   edge is better because it also improves v₃ or reduces effective resistance elsewhere.
   The policy can learn these multi-objective tradeoffs.

3. **Swap mechanism**: FV builds from ring, can't undo. REFINE+ starts with m edges and
   swaps — it can remove structurally redundant edges (high cn, low R) and add high-impact
   ones. This is fundamentally more powerful than constructive placement.

4. **Eigenvalue crossing awareness**: When λ₂ ≈ λ₃, the Fiedler direction is unstable.
   FV doesn't know this and may optimize the wrong bottleneck. REFINE+ sees feat[6]
   (spectral gap) and can learn to stabilize before optimizing.

---

## Policy Network (~20K params)

Same MLP architecture, wider input:
- **ADD MLP**: (10 edge features) → 64 → 64 → 1 logit
- **REM MLP**: (10 edge features) → 64 → 64 → 1 logit
- **Value MLP**: (7 graph features) → 64 → 64 → 1 scalar

---

## Training Strategy

- **Per-swap exact rewards**: At training sizes (n=8-16), O(N³) eigensolve per swap is cheap.
  Each swap gets its own Δλ₂ reward — 120× better credit assignment than macro-step.
- **M-scaled swaps**: num_swaps = max(1, int(m * swap_frac)), swap_frac=0.1.
- **Full eigendecomposition per K-step**: Not just v₂,v₃ but all eigenvalues for computing R, λ₄.
- **PPO with GAE**: Same optimizer setup as REFINE v7.

---

## Implementation Plan

1. [x] Add `compute_effective_resistance()` to refine_env.py
2. [x] Add `compute_common_neighbors()` to refine_env.py
3. [x] Add `get_edge_level_features_rich()` to refine_env.py
4. [x] Create `models/refine_p_policy.py` with 10-dim input
5. [ ] Create `train_refine_p.py` with per-swap rewards
6. [ ] Create `eval_refine_p.py`
7. [ ] Train on n=8-16, 50k episodes
8. [ ] Evaluate at n=8-16, then n=64, 128
