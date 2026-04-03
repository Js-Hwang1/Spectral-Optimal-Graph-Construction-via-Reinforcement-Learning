# BUILD.md -- Architecture for Saddle-Aware RL Graph Rewiring

---

## Overview

Two agents operating on a greedy edge-swap trajectory for lambda2 maximization:

1. **Climber Agent**: scores edge swaps to climb toward local optima
2. **Navigator Agent**: monitors the climb, detects saddles, decides branching

The system inferences greedily. The navigator passively observes the climber's score distribution and intervenes only at saddle points (saving state) and leaf nodes (triggering backtrack).

---

## Landscape Model

Every state during inference is one of three types:

- **PATH**: the gradient (climber's scores) is confident. Just a passing state. Follow greedy.
- **SADDLE**: the gradient is ambiguous (multiple candidates score similarly). A branching point where different greedy choices lead to different local optima. Save state, follow one branch.
- **LEAF**: the greedy policy has entered a cycle (same edges added/removed repeatedly). A local optimum. Backtrack to most recent saved saddle and try a different branch.

Detection:
- PATH vs SADDLE: the navigator classifies based on score distribution features
- LEAF: cycle detection. Compare current epoch's add/rem edges with previous epoch. If identical, leaf detected. O(1) check.

---

## Agent 1: Climber

### Purpose
Score all candidate edges for ADD and REM to make the greedy swap decision at each step.

### Architecture
Shared node embedder + 2 scorer heads (ADD and REM), same as current test.c design:

```
Node embedder: NE_IN -> H -> H (SiLU)
ADD scorer:    H -> H -> 1 (scores which edge to add)
REM scorer:    H -> H -> 1 (scores which edge to remove)
Target scorer: H+extra -> H -> 1 (scores target given source)
```

H = 64 (hidden dimension).

### Node Features (all scale-invariant, normalized by n)

| # | Feature | Formula | Init | Update |
|---|---------|---------|------|--------|
| 0 | Degree | deg[i] / (n-1) | O(N) | O(1) |
| 1 | Step in phase | s / swaps | O(1) | O(1) |
| 2 | Triangle coeff | tri[i] / C(deg[i],2) | O(N^3) | O(N) |
| 3 | SND | snd[i] / (n-1)^2 | O(N^2) | O(N) |
| 4 | Target density | (m-(n-1)) / (max_m-(n-1)) | O(1) | static |
| 5 | Fiedler component | n * v2[i]^2 | O(N^2) Lanczos | O(N^2) warm |
| 6 | ER centrality | n * L+[i,i] | O(N^3) init | O(N^2) Sherman-Morrison |

NE_IN = 7.

### Target Scorer Features

| # | Feature | Formula | Cost |
|---|---------|---------|------|
| 0-63 | Target embedding | H-dim from shared embedder | O(H^2) |
| 64 | Common neighbors | cn[src,tgt] / n | O(1) lookup |
| 65 | Fiedler gap | n * (v2[src] - v2[tgt])^2 | O(1) |
| 66 | Effective resistance | n * R_eff(src, tgt) | O(1) from L+ |

TS_IN = H + 3.

### Operation per step
1. Warm Lanczos -> v2 (O(N^2))
2. Sherman-Morrison L+ update (O(N^2))
3. Node embed all (O(N * H^2))
4. Score ADD candidates -> softmax -> greedy pick (O(N))
5. Apply ADD, update features (O(N))
6. Score REM candidates -> softmax -> greedy pick (O(N))
7. Apply REM, update features (O(N))

Total per step: O(N^2) dominated by Lanczos and Sherman-Morrison.

### Training
**No AZ sweep.** Train on the actual 1-swap sequential landscape:

At each step during training (n=8-12):
- Agent scores all candidates
- For each candidate, compute exact lambda2 of the resulting graph (eigensolve)
- Target distribution = softmax(exact_lambda2 / tau) over candidates
- Loss = KL divergence between agent's softmax and target distribution

This is O(N^2 * N^3) = O(N^5) per step at training time (N^2 candidates, each needing N^3 eigensolve). At n=12: 66 candidates * 12^3 eigensolve ~ 100K ops per step. Feasible.

Training data: (n, m) pairs for n=8-12, all valid m. Use enumerated OPT (data/OPT_{n}.csv) as the reward upper bound.

---

## Agent 2: Navigator (Saddle Detector + Branch Selector)

### Purpose
Passively monitor the climber's score distribution. Make two decisions:
1. Is this state a saddle? (binary: save state or not)
2. If saddle: which branch should be explored after backtracking? (ranking over tied candidates)

These are merged into a single agent that outputs a per-candidate "saddle score." If the max saddle score exceeds a learned threshold, the state is classified as a saddle and the candidate ordering determines branch priority.

### Input Features (all scale-invariant)

**Score distribution features (from Climber's output):**

| # | Feature | Formula | Notes |
|---|---------|---------|-------|
| 0 | Score CoV | std(scores) / mean(scores) | Coefficient of variation. High = confident, low = ambiguous |
| 1 | Score CoV derivative | (CoV_t - CoV_{t-1}) / CoV_t | Relative change. Negative = becoming more ambiguous |
| 2 | Relative gap (1st-2nd) | (s[0] - s[1]) / s[0] | How dominant is the best candidate |
| 3 | Entropy ratio | entropy(softmax(scores)) / log(n_candidates) | 0 = peaked, 1 = uniform |
| 4 | Entropy derivative | (ent_t - ent_{t-1}) / ent_t | Getting more/less ambiguous |
| 5 | Fraction improving | n_improving / n_total | What fraction of candidates improve lambda2 |

**Graph-level features:**

| # | Feature | Formula | Notes |
|---|---------|---------|-------|
| 6 | Lambda2 / n | Lanczos estimate | How good is current graph |
| 7 | Spectral gap | (lambda3 - lambda2) / lambda2 | Fiedler vector stability |
| 8 | Mean degree / (n-1) | Global density proxy | |
| 9 | Degree std / mean_degree | Degree regularity | |

**Per-candidate features (for branch ranking):**

| # | Feature | Formula | Notes |
|---|---------|---------|-------|
| 0 | Net FV score | add_gap - rem_gap | Same as climber's score |
| 1 | Add gap | n * (v2[ai] - v2[aj])^2 | Scale-invariant |
| 2 | Rem gap | n * (v2[ri] - v2[rj])^2 | Scale-invariant |
| 3 | Add R_eff | n * R_eff(ai, aj) | From L+ |
| 4 | Rem R_eff | n * R_eff(ri, rj) | From L+ |
| 5 | Score rank / n_tied | Position among tied candidates | Normalized |

### Output
Per-candidate scalar: saddle_score. Interpretation:
- max(saddle_score) > threshold -> state is a SADDLE, save it
- Candidates sorted by saddle_score descending = branch exploration order

### Training
Binary + ranking supervision:

At each step during training (n=8-12):
1. Run the climber's greedy trajectory
2. At states where score CoV is low (potential saddle), run ALL tied branches to their leaves (greedy to convergence, using exact lambda2)
3. Label the state as SADDLE if different branches lead to different leaves (lambda2 differs by > epsilon)
4. For saddle states, label the correct branch as the one reaching the highest leaf

Loss:
- Binary cross-entropy for saddle/not-saddle classification
- Ranking loss (e.g., margin ranking or listwise) for branch ordering at saddle states

Training cost per episode: O(n_saddles * n_tied * climb_length * N^3). At n=12 with ~5 saddles per trajectory, ~4 tied branches each, ~10 climb steps each: 5 * 4 * 10 * 12^3 ~ 350K eigensolves. ~0.1 seconds on CPU. Very feasible.

---

## Inference Algorithm

```
Input: (n, m, trained_climber, trained_navigator)
Init: random spanning tree + random edges
      init tri/snd/cn: O(N^3)
      compute L+: O(N^3) eigendecomposition
      warm Lanczos -> v2: O(N^2)

Stack = empty
best_graph = init_graph
prev_edges = None

For step = 1 to budget (4*N epochs, 1 swap per epoch):

    1. Climber scores all ADD candidates
       ADD_scores = softmax(climber_add(features))
       
    2. Navigator evaluates score distribution
       nav_features = [CoV, CoV_deriv, rel_gap, entropy, ...]
       per_candidate_saddle_scores = navigator(nav_features, candidate_features)
       
       if max(saddle_scores) > threshold:
           PUSH(stack, current_graph_state, candidate_ranking)
       
    3. Greedy ADD: pick argmax(ADD_scores)
       Apply ADD, update tri/snd/cn, Sherman-Morrison L+, warm Lanczos
       
    4. Climber scores all REM candidates
       Greedy REM: pick argmax(REM_scores)
       Apply REM, update tri/snd/cn, Sherman-Morrison L+, warm Lanczos
    
    5. Cycle check (leaf detection):
       if current add/rem edges == prev_edges:
           cycle_count++
           if cycle_count >= 2:
               Record lambda2, update best_graph if improved
               POP stack -> restore state, get next branch to try
               if stack empty: break
               cycle_count = 0
       else:
           cycle_count = 0
       prev_edges = current add/rem edges

Return best_graph
```

### Complexity
- Per step: O(N^2) for Lanczos + O(N^2) for Sherman-Morrison + O(N) for scoring
- Total: O(N * N^2) = O(N^3) for 4N epochs
- Stack memory: O(saddles * N^2) -- negligible

---

## Training Plan

### Phase 1: Climber
- Train on n=8-12, all valid m
- 1-swap exact lambda2 evaluation per candidate
- KL divergence loss against exact lambda2 softmax targets
- Validate against OPT data (data/OPT_{n}.csv)

### Phase 2: Navigator
- Freeze climber weights
- Run climber's greedy trajectories on n=8-12
- At low-CoV states, exhaustively evaluate branches to label saddles
- Train binary classification + branch ranking
- Validate: does navigator + climber find OPT more often than climber alone?

### Phase 3: Joint fine-tuning (optional)
- Unfreeze both, train end-to-end with lambda2 reward
- Risk of catastrophic forgetting -- may skip this

---

## Evaluation Plan

### Ground truth comparison (n=8-12)
- Exact OPT from enumeration (data/OPT_{n}.csv)
- Report: fraction of (n,m) configs where the agent finds OPT
- Compare: greedy-only vs saddle-aware

### Baseline comparison (n=16-64)
- FV greedy, ER greedy, SW baselines (data/*.csv)
- Report: mean lambda2 / baseline lambda2 across all (n,m)
- Density breakdown by decile

### Ablations
- Climber only (no navigator): how much does hill-climbing quality matter?
- Navigator only (with FV greedy climber): how much does saddle detection matter?
- Full system vs fixed-tau saddle DFS (our C heuristic): RL vs hand-tuned
- Per-swap Lanczos vs per-epoch Lanczos: feature freshness impact

### Scaling test
- Train on n=8-12, evaluate zero-shot on n=16, 32, 64, 96
- Key question: does the saddle detection generalize? (threshold tau is density-dependent, not n-dependent -- our PoC suggests yes)

---

## Open Design Decisions

1. **Shared vs separate embedders.** Should climber and navigator share the node embedder? Sharing reduces parameters and ensures consistent graph representation. Separate allows specialization.

2. **How many branches to save per saddle.** Cap at 6 performed best empirically. Could be a learned parameter.

3. **Budget allocation.** Fixed 4N steps total. No explicit split between climbing and backtracking -- the cycle detector naturally handles this. But should the navigator be more conservative early (save fewer saddles) and more aggressive later (save more)?

4. **Bidirectional descent.** Deferred. Only needed for extreme sparse/dense configs where analytical solutions exist. Can add later as Agent 3 if needed.

5. **Sherman-Morrison numerical stability.** After many rank-1 updates, L+ may drift. Periodic refresh (full eigendecomposition every K steps) may be needed. Cost: O(N^3) every K steps. If K = N, total refresh cost is O(N^2 * N) = O(N^3). Fits budget.
