# CRL Findings -- Empirical Analysis of RL for Algebraic Connectivity

**Date:** 2026-03-31

---

## Problem

Given (n, m), construct a graph with n nodes and m edges maximizing algebraic connectivity (lambda2). Train at n=8-10, generalize to n=64+. Inference must be O(N^3).

---

## 1. The Convergence Crisis

### Symptom
The agent improves lambda2 for 5-15 epochs, then monotonically degrades over the remaining 95% of epochs. best_l2 reached 102% of baselines but final_l2 was 70%.

### Root Cause: Degeneracy Cycling
The agent enters a **3-edge cycle** -- adds edges {A, B, C} then removes the exact same {A, B, C} every epoch. Lambda2 stays constant forever.

**Why it cycles:**
- ADD head scores are near-uniform (entropy ratio 0.999, std 0.008 at n=32)
- Argmax picks the same node deterministically every epoch
- REM head undoes whatever ADD did
- Features (degree, tri, snd, cn) don't provide enough spectral information to differentiate edges on optimized graphs

**Evidence:** PoC/entropy_trace.py traces showed add_std flatlines within 5-10 epochs. Edge tracking confirmed identical add/remove sets every epoch.

### Tabu Made It Worse
Tabu was designed to break cycles by banning add-then-removed edges for n epochs:
- Banning cycled edges forces ADD to pick alternatives
- But ADD has no signal (near-uniform scores) -- alternatives are random
- Random alternatives also get cycled and banned
- **Cascade**: after n/3 epochs, ALL non-edges are banned
- ADD produces 0 edges, REM still removes 3 -- net edge loss -- lambda2 crashes

**Evidence:** PoC traces showed epochs 16-18 with 0 adds but 3 removes each. Lambda2 crashed from 4.14 to 2.68.

**Decision: tabu disabled permanently.**

---

## 2. Feature Analysis

### ADD vs REM Signal (PoC/feature_signal_poc.py)

Spearman rank correlation of features with delta-lambda2:

| Feature | ADD (random init) | ADD (optimized) | REM (random init) | REM (optimized) |
|---------|------------------|-----------------|-------------------|-----------------|
| cn_ij | -0.50*** | -0.04 (dead) | +0.34 | +0.82*** |
| deg_sum | -0.17 | -0.20 | +0.38** | +0.63*** |
| snd | -0.33*** | -0.18 | +0.29* | +0.63*** |

**ADD signal collapses on optimized graphs. REM signal stays strong throughout.**

The initial improvement (epochs 1-5) comes from the ADD head's brief signal window on fresh random graphs (degree-balancing). After that, ADD is blind.

### Spectral Features (Fiedler Vector)
Adding v2_sq (node) and fv_gap (target) from warm Lanczos:
- n=32: 70.3% -> 101.7% vs Best
- n=64: 70.9% -> 100.2% vs Best

Per-swap Lanczos refresh (update v2 after every edge change) adds +1.2% at n=32.

### Effective Resistance
Ablation on n=32 (PoC/ablation_decisions.py), 4N epochs, greedy, no MLP:

| Decision Method | vs FV | vs Best |
|----------------|-------|---------|
| Random | 67.3% | 66.6% |
| FV greedy | 90.6% | 89.2% |
| **ER greedy** | **99.6%** | **98.3%** |

ER captures pairwise global connectivity. FV only captures one spectral cut. ER greedy nearly matches baselines without any learning.

### Epoch Progress Feature
Removed. Trained at n=8-10 where episodes have 8-30 epochs. At n=32 inference with 400+ epochs, epoch_progress barely moves past 0.02. Useless feature that wasted parameters.

---

## 3. Epoch Count Scaling

### Old: K = C * (max_m - m)
At n=32 m=40: 456 epochs. Agent cycles at epoch 7. 449 epochs wasted.

### New: K = C * N
With C=4: 128 epochs at n=32. The agent's productive window is ~10 epochs regardless of m. Scaling with N (not M) matches the actual useful computation.

---

## 4. Exploration Mechanisms Tested

### Stochastic Inference (Softmax Sampling)
Replaced argmax with softmax sampling at inference. **Zero effect** -- the softmax distribution is so peaked that sampling equals argmax. The policy is overconfident.

### Epsilon-Greedy
Random node selection with probability epsilon. Breaks cycles. +1-2% vs Best. But without Metropolis, gains are lost to subsequent bad moves.

### Adaptive Swap Promotion (3 -> 4 -> 5)
Detect degeneracy via add_std flatline. Promote swap count to break cycle:
- swaps=3: cycles
- swaps=4: often still cycles
- swaps=5: breaks out, oscillates productively
- swaps=6: degenerates into new cycle

Best: promote to max 5, don't reset to base when climbing.

### Metropolis (Lanczos lambda2)
Accept/reject epochs based on lambda2 change from warm Lanczos. Acts as ratchet -- only lets lambda2 go up. Combined with promotion at level 5: locks in gains from promoted exploration.

### SA Temperature Cooling
Geometric cooling T_start -> T_end. Early epochs accept worsening moves (explore), late epochs reject (exploit). Helps at n=32 (+1-2%) but gains don't scale to n=64 without T scaling with 1/n.

### Random ADD When Degenerate
Replace MLP ADD with random when degeneracy detected. +0.5% on top of other mechanisms. The MLP ADD has no useful opinion during degeneracy -- random is equivalent.

---

## 5. Rewire Landscape Analysis

### Enumeration (n=6, m=7)
- 5700 connected graphs, 19 non-isomorphic
- 146940 rewire edges (Hamming distance 2 in edge space)
- **2 local optima levels**: lambda2=1.268 (global) and lambda2=1.000 (79% of global)
- Landscape is non-convex with distinct basins

### Key Insight
The local gradient (Fiedler gap) points to whichever peak has steeper local ascent. It cannot distinguish local from global optima. No local signal can tell you "the global optimum is that way."

---

## 6. Saddle Point Escape (PoC/saddle_escape.py)

### Algorithm
1. **ASCEND**: greedy FV-guided rewire until local optimum
2. **DESCEND**: take random/informed worsening rewires, monitor v2 correlation with peak
3. **DETECT**: when |dot(v2_current, v2_peak)| < threshold, basin has changed
4. **REPEAT**: ascend in new basin, keep best peak

### Results

| Method | n=8 vs Best | n=16 vs Best |
|--------|-------------|-------------|
| Pure greedy (1 ascent) | 98.6% | 90.6% |
| Saddle escape (30 rounds, corr=0.7) | 108.2% | 101.9% |

### Hyperparameters
- **corr_threshold=0.7** beats 0.5 and 0.3. Shallow escapes explore nearby basins (good basins cluster together). Deep escapes land randomly.
- **max_rounds=30** saturates for n=8. More rounds don't help once all reachable basins are explored.

### Informed Descent
Tested anti-gradient descent strategies (n=8, 30 rounds):

| Descent Strategy | vs Best |
|-----------------|---------|
| FV-ANTI (disrupt bottleneck) | 103.7% |
| RANDOM | 106.5% |
| **ER-ANTI (restructure connectivity)** | **107.3%** |

ER-ANTI wins: remove high-R_eff (critical) edges, add low-R_eff (redundant) ones. Deliberately restructures global connectivity rather than random or local bottleneck disruption.

---

## 7. Two-Level Optimization Framework

### Level 1: Global Navigation (Physics-Informed)
Use structural properties of good graphs as a compass:
- Near-regular degree sequences (all degrees within 1 of 2m/n)
- Small diameter
- High vertex connectivity
- No small edge cuts

This restricts search to the right REGION of graph space.

### Level 2: Local Search + Basin Escape
Once in the right region:
- FV/ER gradient for local ascent (climb to nearest peak)
- Saddle escape for basin hopping (v2 correlation detector)
- ER-ANTI descent for informed escape direction

### RL Opportunity
The descent phase is currently random or heuristic. An RL agent could learn:
- WHICH direction to descend (which edges to swap during escape)
- WHEN to stop descending (learned basin detector vs fixed threshold)
- HOW to predict which neighboring basin is likely better

---

## 8. Performance Summary

### Best Results (no RL, pure heuristic)

| Method | n=8 | n=16 | n=32 | n=64 | n=96 |
|--------|-----|------|------|------|------|
| Saddle escape (30r, corr=0.7) | 108.2% | 101.9% | -- | -- | -- |
| MLP spectral + promote + metro | -- | ~103% | 102.3% | 100.3% | 97.6% |
| ER greedy (no MLP) | -- | -- | 98.3% | -- | -- |
| FV greedy (no MLP) | -- | -- | 90.6% | -- | -- |
| Random | -- | -- | 66.6% | -- | -- |

### Baselines

| Method | Description | Strengths |
|--------|-------------|-----------|
| FV | Fiedler vector scoring | Good local gradient |
| ER | Effective resistance | Good global redundancy |
| SW(p) | Small-world rewiring | Simple but weak |
| GA | Genetic algorithm | Global search via crossover, slow |

### Baseline Bug
SW baselines report impossible lambda2 values at extreme densities (e.g., lambda2=6.8 for n=8 m=27 when maximum possible is 6.0). Baselines need re-generation.

---

## 9. Key Files

| File | Purpose |
|------|---------|
| src/test.c | Main AZ-style RL (train + inference) |
| src/ga_main.c | Genetic algorithm (optimized with SM + bridges) |
| PoC/feature_signal_poc.py | Feature-lambda2 correlation analysis |
| PoC/entropy_trace.py | Per-epoch trace of scores/entropy/decisions |
| PoC/ablation_decisions.py | Random vs FV vs ER greedy comparison |
| PoC/rewire_landscape.py | Enumerate all graphs, visualize landscape |
| PoC/rewire_landscape_canonical.py | Non-isomorphic landscape visualization |
| PoC/saddle_escape.py | Saddle point escape PoC (n=8) |
| PoC/saddle_sweep_n16.py | Saddle escape sweep (n=16) |
| PoC/saddle_informed.py | Informed descent comparison |
| data/ablation_n32.csv | Raw ablation data |
| data/saddle_n16.csv | Saddle escape results n=16 |

---

## 10. Open Questions

1. **Saddle escape at scale**: Does the v2 correlation detector work at n=64+? How many rounds needed?
2. **RL for descent**: Can an agent learn better escape directions than random/ER-ANTI?
3. **Spectral gap (lambda3 - lambda2)**: Unused signal. When gap is small, FV is unreliable -- should switch strategy.
4. **Combining ER + FV**: ER for global navigation, FV for local refinement. Adaptive switching.
5. **Training at larger n**: Current training at n=8-10. Would n=16-24 training improve generalization?
6. **Sherman-Morrison L+ at inference**: O(N^2) per edge vs O(N^2) for Lanczos. Same budget, richer signal.
