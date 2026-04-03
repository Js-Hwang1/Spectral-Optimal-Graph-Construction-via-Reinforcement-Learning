# CRL Development Plan

## Problem

Given (n, m), construct a graph with n nodes and m edges that maximizes algebraic connectivity λ₂. Train at n=8–24, generalize to n=256+. Inference must be O(N³) or better.

---

## Breakthrough: GNN RL with Autoregressive Removal

### Algorithm

Single epoch:
1. Start with random G(n, m) (ring + random edges)
2. Add m random edges (union with random set) → G(n, ~2m)
3. GNN autoregressively removes edges one-by-one back to m:
   - At each step: GNN sees full graph state, scores all removable (non-bridge) edges
   - Softmax sample one edge to remove
   - Dense reward via eigensolve: Δλ₂ = λ₂_after - λ₂_before (training only)

### Architecture

GNN Policy (21K params):
- 2-layer GCN (node_feat=1 → hidden=64 → hidden=64)
- Node input feature: degree/(n-1)
- Edge scorer: MLP([h_i || h_j || global_mean_pool]) → score
- Value head: MLP(global_mean_pool) → value
- Training: REINFORCE with learned baseline

### Results

**BUG FOUND AND CORRECTED:** The initial "125% of baselines" result was due to a bug in `run_episode` — it returned `best_l2` which was the maximum λ₂ seen during removal, including intermediate graphs with MORE than m edges. A graph with ~2m edges naturally has higher λ₂ than an m-edge baseline. The comparison was apples-to-oranges.

**Corrected results (final m-edge graph λ₂):**

| Method | n=8 vs FV | n=16 vs FV | n=24 vs FV |
|--------|-----------|------------|------------|
| FV baseline | 100% | 100% | 100% |
| GNN RL autoregressive, add=m | 96% | 94% | 94% |
| GNN RL autoregressive, add=N | 92% | 83% | 83% |
| GNN RL batched K=N, add=m | 54% | 22% | 13% |

The GNN RL approach UNDERPERFORMS baselines. The autoregressive framework is interesting (94% from a learned policy with no spectral features) but doesn't beat FV.

### Validation (post-correction)

- Edge count: final_m == target_m ✓
- Connectivity: all graphs connected ✓
- λ₂: independently verified ✓
- **Bug:** `best_l2` in `run_episode` tracked MAX over all intermediate states (including over-edged graphs). Fixed by reporting `final_l2` only.

### What We Learned

1. **Autoregressive removal is viable** — the GNN learns to make reasonable removal decisions from raw graph structure (94% of FV with no spectral features). This is a proof of concept that GNN + RL can operate on this problem.

2. **Dense eigensolve reward works** for training — each removal gets exact Δλ₂ feedback.

3. **Batching destroys quality** — batched removal (22%) vs autoregressive (94%). The GNN needs to see the graph after each removal.

4. **More adds help** — add=m (94%) >> add=N (83%). The random search space matters.

5. **94% is below baselines** — the GNN alone can't match FV scoring. FV uses exact spectral information; the GNN has to learn it from scratch with only degree features.

6. **ALWAYS verify the metric.** The `best_l2` bug inflated results by comparing intermediate over-edged graphs against correct-edge baselines. Track and report only the FINAL graph's λ₂.

---

## Time Complexity Analysis

### Current (single epoch)

| Operation | Per-removal | Removals | Total |
|-----------|------------|----------|-------|
| adj_to_pyg | O(N²) | ~m ≈ N² | O(N⁴) |
| GNN forward (2 GCN layers) | O(N²) | N² | O(N⁴) |
| find_bridges | O(N²) | N² | O(N⁴) |
| score_edges | O(N² × H) | N² | O(N⁴) |
| eigensolve (training only) | O(N³) | N² | O(N⁵) |

**Training: O(N⁵) per episode** — acceptable (n≤24, unrestricted time).
**Inference: O(N⁴) per episode** — over budget (need O(N³)).

### Bottleneck

O(N²) autoregressive removal steps × O(N²) per step = O(N⁴).

### Path to O(N³)

Options (not yet tested):
1. **Add O(N) edges instead of O(N²):** fewer removals needed → O(N) steps × O(N²) = O(N³)
2. **Batch removal:** GNN scores all edges once, remove bottom-k in one shot → O(1) GNN calls × O(N²) = O(N²)
3. **Multi-epoch with small adds:** K epochs × O(√N) adds per epoch → K×√N removals. If K=O(√N): O(N) removals × O(N²) = O(N³)
4. **Non-autoregressive:** one-shot edge scoring without sequential removal. Loses the autoregressive advantage but gains O(N²) total.

---

## Next Steps (prioritized)

### 1. Multi-epoch iteration
Current results are from a SINGLE epoch (one add-remove pass). Multiple epochs should improve quality further — each epoch sees a different graph state and gets another chance to refine.

### 2. Reduce add count for O(N³)
Add O(N) random edges instead of m ≈ O(N²). This reduces removals to O(N), achieving O(N³) inference. Test whether the quality holds with fewer adds.

### 3. Scale testing
Test at n=32, 64 to verify the GNN generalizes beyond n=24.

### 4. Port to C
Once the design is validated, port the GNN (forward only) to C for production inference. The GNN forward pass is just matrix multiplications + message aggregation — BLAS-friendly.

---

## Previous Findings (for reference)

### GA
- FV-guided crossover (union + spectral pruning) + FV-guided mutation
- Bridge-accelerated pruning: O(N²) per crossover (lossless, +1-4% vs BFS)
- Lanczos fitness: O(N²) per eval (lossless vs eigensolve)
- Pop=1.5N, gen=40, mut=2 gives 101-114% of baselines
- Degradation with n due to population budget (not ceiling — unrestricted GA reaches 108% at n=32)
- Population is the scaling bottleneck, not generations or mutations
- More mutations HURT (destroy diversity for crossover)

### SA
- FV-biased proposals + Metropolis acceptance
- 103-106% of FV at n=16-24 with iter_mult=10
- O(N⁵) cost — not viable at scale

### Refine (dual-phase)
- Batched add → Lanczos → batched remove with Metropolis
- 97-99% of baselines with Metropolis, 94-98% without
- Edge-level RL (Patch 1) failed: FV gap is already optimal for single-graph scoring
- Tabu search: hurt at all settings

### Key insight
The FV gap is a near-optimal LOCAL scoring signal. The GNN provides global awareness via message passing, but at 94% vs FV it doesn't surpass the spectral signal. The best approach may be combining GNN (global structure) with FV (spectral precision) — giving the GNN access to Fiedler features rather than just degree.

---

## Files

### Source (src/)
- `crl.h`, `crl.c` — Core library (Lanczos, bridges, MLP, PPO)
- `main.c` — Training binary (dual-phase PPO)
- `eval_main.c` — Eval binary (--metropolis, --lanczos-metro, --tabu, --ref-augmented, --ga)
- `ga_main.c` — GA binary (crossover + mutation, OpenMP parallel)
- `sa_main.c` — SA baseline
- `v10_main.c` — Node-centric greedy

### PoC (PoC/)
- **`gnn_rl_poc.py`** — GNN RL breakthrough (PyTorch + PyG), 126% of FV
- `ga_refine_poc.py` — GA Python PoC
- `dual_phase_refine_poc.py` — Dual-phase no-RL baseline
- `ref_augmented_poc.py` — Reference-augmented RL
- `rr_drift_poc.py` — RR drift analysis

### Documentation
- `crl_ga.md` — GA algorithm details and results
- `CLAUDE.md` — Project instructions and current algorithm
