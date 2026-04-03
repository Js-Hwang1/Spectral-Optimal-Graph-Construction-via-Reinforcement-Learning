# Navigator Training Log

## Current Design (v6 -- Detector + GOD Ranking)

### Architecture
- MLP: 4 -> 16 -> 16 -> 1 (SD_NAV_IN=4, SD_NAV_NH=16, ~337 params)
- Activations: SiLU (hidden), sigmoid (output)
- Output: logit. sigmoid(logit) > 0.5 means "saddle detected"
- Inference threshold tunable via --nav-threshold (in logit space)

### Input Features
| # | Name | Formula | t-stat (n=16) | t-stat (n=24) |
|---|------|---------|---------------|---------------|
| 0 | entropy | softmax entropy of ALL add scores / log(n_add) | -11.0 | -13.9 |
| 1 | l2_n | lambda2 / n | -7.4 | -6.7 |
| 2 | rho | (m - (n-1)) / (n(n-1)/2 - (n-1)) | -4.1 | -1.9 |
| 3 | spec_gap | (lambda3 - lambda2) / n | +7.6 | +14.6 |

All features computable from GraphState. No per-candidate features (detection only, not ranking).

### Label
"Do branches diverge?" -- at each training step:
1. Find top-6 ADD candidates by climber score
2. Clone graph 6 times, apply each candidate + greedy REM, roll out to stagnation
3. Record 6 leaf lambda2 values
4. Label = 1 if max - min > 0.01, else 0
5. ~80% positive rate at n=8-16 (most steps have some divergence)

### Training Procedure
1. Load frozen climber (50k episodes, edge-level KL training)
2. Per episode:
   - Sample n in [min_n, max_n], m clipped to rho in [0.1, 0.8]
   - Build ring + random graph, run greedy climber trajectory
   - At each step: compute features, label via 6-branch rollout, BCE loss
   - Accumulate gradients across steps
3. Adam update once per episode (lr=1e-3, grad_clip=1.0, weight_decay=1e-4)
4. 10k episodes, n=8-16, ~98k training examples

### Inference (episode.c)
- Each step: compute features, forward MLP
- logit > nav_threshold -> SADDLE: build top-6, GOD-style exhaustive rollout, pick best, push DFS stack
- logit <= threshold -> GREEDY: climber's top-1 pick, no branching
- On stagnation (lambda2 unchanged 3 steps): backtrack from DFS stack

### Results (v6, 4 features, divergence label, m-clipped)

**Threshold sweep at n=16:**
| Threshold | Fire rate | vs Best | GOD gain captured |
|-----------|-----------|---------|-------------------|
| 0 | 83% | 105.6% | 105% |
| 1 | 71% | 104.8% | 92% |
| 2 | 51% | 102.9% | 60% |
| 3 | 18% | 100.9% | 27% |
| GOD | 100% | 105.3% | 100% |
| Climber | 0% | 99.3% | 0% |

**Across n values (v6):**
| n | thresh=0 | thresh=1 | thresh=2 | thresh=3 | GOD | Climber |
|---|---------|---------|---------|---------|-----|---------|
| 8 | 102.8% | 102.8% | 103.9% | 102.9% | 103.6% | 101.6% |
| 10 | 104.9% | 104.7% | 104.8% | 102.3% | 105.1% | 100.4% |
| 12 | 105.6% (86%) | 105.2% (70%) | 104.9% (50%) | 102.7% (6%) | 105.5% | 101.0% |
| 14 | 105.2% (76%) | 105.0% (51%) | 103.9% (26%) | 101.1% (2%) | 104.6% | 99.9% |
| 16 | 105.6% (83%) | 104.8% (71%) | 102.9% (51%) | 100.9% (18%) | 105.3% | 99.3% |
| 24 | 102.4% (89%) | 100.9% (74%) | 97.8% (61%) | 94.7% (27%) | 102.5% | 93.9% |

## GOD Saddle Analysis

From instrumented GOD episodes -- when does branching actually help?

**Non-greedy win rate (GOD picks different from greedy, gain > 0.01):**
| n | Greedy OK | Non-greedy wins | Avg gain |
|---|-----------|----------------|----------|
| 8 | 94.0% | 6.0% | 0.165 |
| 10 | 88.8% | 11.2% | 0.273 |
| 12 | 86.9% | 13.1% | 0.215 |
| 14 | 83.4% | 16.6% | 0.147 |
| 16 | 85.4% | 14.6% | 0.219 |
| 24 | 88.5% | 11.5% | 0.253 |

Only ~12% of steps are real saddles where branching changes the outcome.

**Feature separation (real saddle vs greedy-ok):**
| Feature | t (n=16) | t (n=24) | Saddle value | OK value |
|---------|----------|----------|-------------|----------|
| step_f | -10.9 | -17.9 | 0.018 | 0.037 |
| spec_gap | +7.6 | +14.6 | 0.031 | 0.017 |
| entropy | -11.0 | -13.9 | 0.845 | 0.951 |
| gap1K | +8.2 | +9.8 | 1.075 | 0.660 |
| gap12 | +5.4 | +7.5 | 0.356 | 0.193 |
| l2_n | -7.4 | -6.7 | 0.278 | 0.384 |
| fv_gap | +5.1 | +6.0 | 13.27 | 11.09 |
| rho | -4.1 | -1.9 | 0.404 | 0.473 |
| bridge_f | +1.3 | +0.3 | 0.033 | 0.020 |

Saddles happen early (low step_f), at high spectral gap, low entropy, with wider score spread (gap1K), in sparser graphs.

## Historical Attempts (1-5)

### Attempt 1: Binary Classification (BCE)
Label = "do branches lead to different leaves?" 75% positive. Learned "always saddle." Useless.

### Attempt 2: Stricter BCE
Label = "does any non-greedy branch beat greedy by >0.01?" Only 23% positive. Learned "never saddle."

### Attempt 3: Weighted BCE + bias=0
3x weight on positive class. Marginal improvement, no clean boundary.

### Attempt 4: Value Function (MSE regression)
Label = leaf lambda2/n. RMSE 0.08, branch differences <0.05. Result: +1.6% at n=32.

### Attempt 5: Imitation of GOD
Cross-entropy over branches. Only 805 examples in 1000 episodes. 21% accuracy.

### Attempt 6 (current v3-v6): Dense BCE Detector
Separated detection from ranking. Every step labeled. Works with divergence label.
Key insight: detection + GOD ranking matches full GOD at n=10-14.

## Known Issues

1. **Label mismatch**: 80% of steps labeled positive (branches diverge) but only 12% actually benefit from branching. This causes high false positive rate.
2. **Correct label fails**: training with "non-greedy beats greedy" gives noisy signal because greedy training trajectories differ from GOD-guided inference trajectories.
3. **Features converge at large n**: entropy gap 0.10 at n=8, 0.035 at n=24. Detector loses selectivity at n=16+.
4. **rho is weak**: t=-1.9 at n=24. Candidate for replacement with gap1K (t=9.8).

## Untried Features (ranked by GOD t-stat at n=24)
1. step_f (t=-17.9) -- trajectory progress, not intrinsic graph property
2. gap1K (t=9.8) -- top1-top6 score spread, climber-dependent
3. gap12 (t=7.5) -- top1-top2 score gap
4. fv_gap (t=6.0) -- Fiedler gap of greedy edge

## Key Checkpoints
- Climber: `logs/saddle_prod/climber_final.bin` (50k episodes)
- v3 wide (best perf): `logs/saddle_det_v3_wide/navigator_final.bin` (3 features, divergence label)
- v6 (best selectivity): `logs/saddle_det_v6/navigator_final.bin` (4 features, divergence label)
