# Findings: Simulated Annealing REFINE & Non-Greedy Strategies

## Summary

SA-REFINE (Simulated Annealing with FV-biased proposals) is the **first method to beat all baselines (FV, ER, SW)** on algebraic connectivity maximization — but it requires per-swap eigensolves at inference, making it O(N^5) and incompatible with the O(N^3) inference requirement.

Attempts to transfer the SA insight into the learnable REFINE(RL) policy (which has O(N^3) inference) have **not succeeded**.

---

## What SA-REFINE Does

**Algorithm:** Per iteration:
1. Eigensolve current graph → v₂, λ₂
2. Propose swap: add edge with high FV gap `(v₂[i] - v₂[j])²` via softmax, remove edge with low FV gap (avoiding bridges)
3. Apply swap, eigensolve → λ₂_new, Δλ₂ = λ₂_new - λ₂
4. Metropolis acceptance: always accept if Δλ₂ > 0, else accept with prob `exp(Δλ₂ / T)`
5. Geometric cooling: T decays from T_start to T_end
6. Track best graph seen across all iterations

**Winning config:** T_start=1.0, T_end=0.001, iterations=2×m, proposal_tau=0.01

**Results (5 seeds, best-of-seeds per config):**

| n | vs FV_db | vs Best Baseline | Wins vs FV | Wins vs Best |
|---|---------|-----------------|-----------|-------------|
| 8 | 102.2% | 99.3% | 8/20 | 7/20 |
| 16 | **103.3%** | **101.3%** | 84/104 | 63/104 |
| 24 | **101.7%** | **100.8%** | 228/252 | 172/252 |
| 36 | **101.2%** | **100.5%** | 568/594 | 435/594 |

**PoC:** `CRL/PoC/sa_refine_poc.py`
**C implementation:** `CRL/sa_main.c` (standalone executable `crl_sa`)

---

## Why SA Works (and Greedy Doesn't)

### Oracle Analysis (oracle_rank_poc.py)

**Shocking finding:** The per-step oracle (always pick the swap with highest TRUE Δλ₂) performs **WORSE** than greedy FV in full rollout.

- n=16: FV_ST avg λ₂ = 7.021, Oracle = 6.600 (oracle **loses** by 0.42)
- FV picks the oracle's #1 choice only 30.2% of the time
- But FV captures 94.0% of oracle's per-step Δλ₂
- Per-step oracle in top-3 = 97.8% of oracle Δλ₂, but full rollout: 6.871 (still worse than FV)

**Implication:** FV's "approximation error" is actually beneficial — it provides implicit lookahead. The exact gradient is myopic; the approximate gradient accidentally explores better trajectories.

### Feature Analysis (oracle_features_poc.py)

- v₂ gap has highest correlation with oracle Δλ₂ (Spearman ρ = 0.93)
- Second-order perturbation theory HURTS (rollout: 6.722 vs FV: 7.021)
- v₃ gap is anti-correlated (ρ = -0.30)
- No feature-based scoring beats FV in full rollout
- **Conclusion:** The optimal deviation from FV is PROBABILISTIC, not DIRECTIONAL

### What SA Adds

SA's power comes from three mechanisms:
1. **Per-swap evaluation** — eigensolve after each swap to know if it helped
2. **Accept/reject** — undo bad swaps, keep good ones
3. **Best tracking** — return the best graph seen, not the final one

All three require eigensolves at inference → O(N^5) for dense graphs. This violates our O(N^3) constraint.

---

## Attempts to Transfer SA Insight to REFINE(RL)

### Attempt 1: Learned Eigenvector Weights (spectral_weights_poc.py)

**Idea:** Learn state-dependent weights for eigenvector gaps: `score(i,j) = Σ_k w_k(state) · n · (v_k[i] - v_k[j])²`

**Architecture:** WeightNet (584 params): 9 graph features → 32 (Tanh) → 8 (softmax) → weights for v₂..v₉ gaps. Trained via REINFORCE on n=8-10.

**Result:** Network collapsed to near-fixed weights (w₂ ≈ 0.52, others ≈ 0.07). Did not learn meaningful state-dependent blending. Marginal improvement at best — n=24: 101.0% FV_db vs FV_ST at 100.8%.

**Conclusion:** REINFORCE signal too noisy for such a subtle task. The optimal blend is nearly state-independent anyway (v₂ dominates).

### Attempt 2: Temperature/Noise on REFINE (refine_temp_poc.py)

**Idea:** Add fixed temperature to REFINE's FV-based scoring (softmax sampling instead of argmax).

**Result:** `split_a0.01_r0` best at 97.5% FV (n=16), 92.9% FV (n=24). Adding noise helps slightly but fixed noise is not enough.

**Conclusion:** Fixed temperature doesn't capture the dynamic schedule that SA uses (hot→cold).

### Attempt 3: Best-So-Far Reward Shaping (v10_main.c)

**Idea:** Change the REFINE(RL) per-swap reward from `Δλ₂` to `max(0, λ₂_new - best_λ₂_so_far_in_episode)`. This way:
- Worsening swaps get reward = 0 (not negative) — no punishment for exploration
- Only new episode-highs get positive reward
- GAE propagates positive reward backward to the "bad" swaps that set up the path

**Implementation:** Collect episodes with existing `crl_collect_episode()` (per-swap Δλ₂), then relabel rewards by reconstructing λ₂ trajectory from `initial_λ₂ + cumulative Δλ₂`. No changes to crl.c/crl.h.

**Training:** 200k episodes, n=8-16, swap_frac=0.05, PPO with GAE (γ=0.99, λ=0.95). Best checkpoint at episode 48k (peaked early).

**Results:**

| n | % of best baseline | Wins |
|---|-------------------|------|
| 16 | 95.4% | 5 |
| 24 | 92.6% | 0 |
| 36 | 87.9% | 0 |

**Conclusion:** Worse than standard REFINE(RL). The reward was too sparse — most swaps get 0, starving the gradient signal. The model peaked at 48k episodes and stopped improving. Per-swap Δλ₂ provides much denser feedback.

---

## The Fundamental Mismatch

SA beats baselines because it has **per-swap evaluation and undo** at inference time. These require eigensolves.

REFINE(RL) inference has **no eigensolves** — it must propose good swaps from the start, without knowing if they helped. The MLP learns from per-swap Δλ₂ during training but can't use it at inference.

The SA insight (stochastic deviation from greedy helps) is already present in REFINE(RL) via softmax sampling over MLP logits. The MLP's logit scale controls effective temperature. The step_frac feature enables step-dependent behavior. But without per-swap feedback at inference, the policy can't do what SA does — deliberately accept bad moves and undo them if they don't pay off.

---

## Things NOT Worth Trying (Based on These Findings)

1. **Any scoring that tries to beat FV per-step** — the oracle analysis shows no per-step scoring beats FV in rollout. The deviation must be probabilistic, not directional.

2. **Learning v₃/v₄/higher eigenvector weights** — v₃ gap is anti-correlated with oracle (ρ = -0.30). Higher eigenvectors add noise, not signal.

3. **Second-order perturbation theory** — explicitly hurts (6.722 vs FV 7.021 at n=16).

4. **Sparse rewards (best-so-far, final-only)** — too little gradient signal for PPO to learn. Per-swap Δλ₂ is the right reward density.

5. **Fixed temperature/noise on FV** — marginal gains at best. SA's power is the adaptive schedule + accept/reject, not just noise.

---

## What Might Be Worth Exploring

1. **SA as a training-time augmentation** — use SA to generate "expert" trajectories, then distill into the MLP via behavioral cloning. The MLP learns to mimic SA's decisions without needing eigensolves at inference.

2. **Curriculum from SA** — train on (n,m) configs where SA beats FV most, hoping the MLP learns the specific structural patterns that benefit from non-greedy behavior.

3. **Hybrid inference** — use the MLP for O(N²) batch scoring, but add a few (O(1)) eigensolves at key points during inference to track best graph. Within O(N³) budget for sparse/moderate graphs.

4. **Reward blending** — instead of pure best-so-far (too sparse) or pure Δλ₂ (too greedy), try `α * max(0, Δλ₂) + (1-α) * max(0, λ₂ - best_so_far)` with small α. Clips negative rewards while keeping some per-swap signal.

---

## File References

| File | Description |
|------|-------------|
| `CRL/PoC/sa_refine_poc.py` | SA-REFINE Python PoC (BEATS baselines) |
| `CRL/sa_main.c` | SA-REFINE C implementation (`crl_sa`) |
| `CRL/v10_main.c` | Best-so-far reward training (did not help) |
| `CRL/PoC/oracle_rank_poc.py` | Oracle worse than FV (key finding) |
| `CRL/PoC/oracle_features_poc.py` | No feature beats v₂ gap |
| `CRL/PoC/spectral_weights_poc.py` | Learned eigenvector weights (marginal) |
| `CRL/PoC/refine_temp_poc.py` | Fixed temperature on REFINE (marginal) |
