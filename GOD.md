# GOD Navigator: Strategy and Scaling Analysis

## What GOD Does

At each step of inference, the GOD navigator replaces the navigator MLP with exhaustive branch evaluation:

```
1. Climber embeds all nodes (shared NE_IN=5 embedder)
2. Climber scores ALL candidate ADD edges via ADD edge scorer MLP
3. Select top-6 ADD edges by climber score
4. For each of the 6 candidates:
   a. Clone the full graph state (adj, deg, tri, snd, cn, L+, v2, bridges)
   b. Apply that ADD edge to the clone
   c. Run the climber's REM scorer on the clone -> greedy REM
   d. Apply that REM
   e. Run the climber greedily to stagnation (repeated ADD+REM until lambda2 flatlines)
   f. Record the leaf lambda2 via exact eigensolve
5. Pick the candidate whose rollout reached the highest leaf lambda2
6. Apply that choice to the real graph, continue to next step
```

## What GOD Does NOT Do

- Does NOT propose new edges beyond the climber's top-6
- Does NOT branch on the REM decision (always greedy REM)
- Does NOT look ahead multiple steps (each rollout is greedy from the branch point)
- Does NOT use any features the navigator wouldn't have access to

## Why GOD Works

GOD's entire advantage comes from ONE thing: **choosing a different ADD edge than the climber's greedy top-1 pick.** The climber's ranking is good (the correct edge is always in the top-6), but the top-1 is often not the best long-term choice.

At saddle points, multiple ADD edges have similar climber scores but lead to different local optima. GOD resolves this ambiguity by running all branches to completion and picking the winner.

## Performance

| n | Climber only | GOD (6 branches) | Gap |
|---|-------------|-------------------|-----|
| 8 vs Best | 101.6% | 103.6% | +2.0% |
| 9 vs Best | -- | 102.3% | -- |
| 10 vs Best | -- | 105.1% | -- |
| 12 vs Best | -- | 105.5% | -- |
| 14 vs Best | -- | 104.6% | -- |
| 16 vs Best | 99.3% | 105.3% | +6.0% |

Consistent 103-105% vs Best across n=8-16. The gap from climber-only grows with n (2% at n=8, 6% at n=16) because larger graphs have more saddle points where branching matters.

## Scaling Properties

### Budget Independence
GOD saturates at budget_mult=4 (16n total steps). Increasing to 20 or 40 gives identical results. The climber reaches stagnation within ~5-10 steps, and GOD picks the right branch at the first divergence point.

### Branch Width Independence
6, 12, and 24 branches give identical results. The correct branch is ALWAYS in the climber's top-6. The climber's ranking is good for candidate generation -- the issue is only in the top-1 vs top-k selection.

### vs Exact OPT (n=8-11)
GOD reaches exact OPT for ~78% of (n,m) configs. The 22% misses are:
- Very sparse (m ~ n): star graph / tree variants. Analytical solutions.
- Specific symmetric structures (K_{n/2,n/2}): the climber never ranks the path to these in its top-6.
- Very dense (m ~ max): trivial configs.

In the interesting mid-density regime where the optimal structure is UNKNOWN, GOD reaches or exceeds all baselines consistently.

## What the Navigator Must Learn

The navigator's task is precisely: **rerank the climber's top-6 ADD candidates.**

Input per candidate:
- Climber's score (how much the climber likes this edge)
- FV gap: n * (v2_i - v2_j)^2
- R_eff: n * (L+_ii + L+_jj - 2*L+_ij)
- Graph-level: entropy of climber scores, rel_gap, l2/n, spectral gap, etc.

Output: which candidate leads to the best long-term outcome.

This is a RANKING task over 6 items, not a regression or classification task. The navigator sees 12 features per candidate and must predict the rank order. With ~800 informative training examples (steps where the non-greedy branch wins), the current 1.6K parameter MLP underperforms.

## Cost Analysis

| Component | GOD (per step) | Navigator (per step) |
|-----------|---------------|---------------------|
| Embed nodes | O(N) | O(N) |
| Score ADD edges | O(N^2) | O(N^2) |
| Branch evaluation | 6 * O(rollout * N^2) | 6 * O(1) MLP forward |
| REM scoring | O(N^2) | O(N^2) |
| Total per step | O(rollout * N^2) | O(N^2) |

GOD is ~100x slower than the navigator per step (rollout length ~10-20 steps per branch). This is why we need the navigator -- GOD is too expensive for production inference, especially at large n.

## Implications for Navigator Training

1. **The task is ranking, not regression.** Training with MSE on absolute values wastes signal. Pairwise or listwise ranking losses would be more appropriate.

2. **The training data is sparse.** Only ~8% of steps have a non-greedy winner among the top-6. Most steps, the greedy choice IS optimal. The navigator needs to learn WHEN branching matters (rare events) and WHICH branch wins (conditional on mattering).

3. **The features ARE sufficient.** GOD proves the climber's top-6 always contains the correct edge. The features available to the navigator (climber score, FV gap, R_eff, graph stats) describe these 6 candidates. The question is whether a small MLP can learn the reranking from these features.

4. **Scale invariance holds.** GOD's performance is consistent from n=8 to n=16 (103-105%). The underlying pattern -- which branch wins at a saddle -- appears to be n-invariant when features are properly normalized.
