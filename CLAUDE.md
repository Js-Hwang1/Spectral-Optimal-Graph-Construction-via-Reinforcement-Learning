# CLAUDE.md - G(n,m) Foundational RL Model

## Goal

Build a deep RL model that, given `(n, m)`, constructs a graph with `n` nodes and `m` edges that maximizes **algebraic connectivity (λ₂)**.

**Constraints:**
- **Training:** O(N³) eigensolvers per swap (per-swap Δλ₂ rewards at n=8-16)
- **Inference:** O(N³) total (matching FV/ER baselines). Batch-score + row-update + RR tracking.
- **Target:** Beat FV and ER baselines stored in HuggingFace Database (see `DB_use.md`)
- **Generalization:** Training at n=8 to n=16, must generalize to n=32,64,128,256,512,1024 smoothly.

---

## Hard Rules

### O(N³) Complexity Budget

**Training and inference must be O(N³) or better**, matching the FV/ER baselines we compete against.

- Per-swap eigensolve during training: O(N³) — fine at n=8-16
- Inference: batch-score O(N²) + M/10 row-updates O(N) each = O(N³) for dense graphs
- Lanczos warm-start per K-step: O(N²) — within budget
- RR subspace updates: O(Nk²) per swap — within budget

### Cython Acceleration

Hot-path functions have Cython-compiled versions in `rl/utils/_fast_features.pyx`. The pure Python fallbacks in `spectral.py` are used when Cython is not compiled.

**Build Cython (required for training speed):**
```bash
cd rl
pip install cython
python setup.py build_ext --inplace
```

**After modifying `_fast_features.pyx`, always rebuild:**
```bash
cd rl && python setup.py build_ext --inplace
```

**When adding new features:** Add pure Python version in `spectral.py` first, then mirror in `_fast_features.pyx`, rebuild, and verify both paths produce identical output.

---

## Current Approach

### Architecture: Edge-MLP + Rayleigh-Ritz Subspace Tracking + PPO (REFINE v8)

Instead of building graphs from scratch, we:
1. **Initialize** with ring + random edges (matches FV/ER baseline starting point)
2. **Rewire** edges iteratively to improve λ₂ across K Lanczos snapshots
3. **Track** the spectral subspace via k=8 Rayleigh-Ritz rotation between swaps
4. **Learn** which edges to add/remove using an edge-level MLP on 10-dim spectral features
5. **Train** with per-swap exact Δλ₂ rewards (120× better credit assignment)

### Key Innovation: Per-Swap Rewards + RR Tracking

**Per-swap rewards (training only):** Each swap (add one edge + remove one edge) gets
its own exact Δλ₂ reward via O(N³) eigensolve. At training sizes n=8-16, this is
~10μs per eigensolve. The MLP learns which individual swaps improve λ₂.

**RR subspace tracking (training + inference):** Adding edge (u,v) updates the
Laplacian by rank-1: `L_new = L + zz^T`. Project into tracked k-dim subspace [v₂..v₉]:
```
δ_m = v_m[u] - v_m[v]                    # O(k) gap vector
L_sub = diag(λ₂..λ₉) + δδ^T             # O(k²) projected Laplacian
lams_new, R = eigh(L_sub)                # O(k³) ≈ O(1) for k=8
[v₂..v₉]_new = [v₂..v₉] @ R            # O(Nk²) = O(64N) rotation
```

### Policy Network (~16K params)

- **ADD MLP**: (10 edge features) → scalar logit for adding non-edges
- **REM MLP**: (10 edge features) → scalar logit for removing edges
- **Value MLP**: (9 graph features) → scalar V(s)
- **Training:** PPO with GAE and **per-swap** rewards

### Episode Flow

```
Input: (n, m)
    ↓
Initialize: Ring + random edges → suboptimal graph
    ↓
For k = 0..K-1:                                        (K = 20)
  Lanczos → V=[v₂..v₉], lams=[λ₂..λ₉]                [O(N²), warm-started]
  Batch-score all N² pairs ONCE                         [O(N²)]
  For each of M/10 swaps:
    Sample ADD, add edge, RR-update, row-update logits  [O(N)]
    Sample REM, remove edge, RR-update, row-update      [O(N)]
    *** Per-swap reward = exact Δλ₂ (training only) *** [O(N³)]
    ↓
Output: Final graph with improved λ₂
```

### Edge Features (10-dim, O(1) each)

```
0: n * |v₂ᵢ - v₂ⱼ|²   — Fiedler gap (RR-tracked)
1: n * |v₃ᵢ - v₃ⱼ|²   — v₃ gap (RR-tracked)
2: n * |v₄ᵢ - v₄ⱼ|²   — v₄ gap (RR-tracked)
3: n * |v₅ᵢ - v₅ⱼ|²   — v₅ gap (RR-tracked)
4: degᵢ / (n-1)        — source degree
5: degⱼ / (n-1)        — target degree
6: (λ₃ - λ₂) / n       — spectral gap 2-3
7: (λ₄ - λ₃) / n       — spectral gap 3-4
8: (λ₅ - λ₄) / n       — spectral gap 4-5
9: step / K             — budget awareness
```

### Reward Structure

- **Per-swap:** Exact Δλ₂ after each individual swap (add+remove). Training only, O(N³).
- Each swap is its own RL transition: (state, action, reward).
- At n=16 dense: ~240 eigensolves/episode × 16³ ≈ 1M ops. Negligible.

### Complexity

**Training (n=8-16):**

| Component | Per-call | Calls/episode | Total |
|-----------|----------|---------------|-------|
| Lanczos (warm, k=15) | O(N²) | K=20 | O(N²) |
| Batch-score MLP | O(N²) | K=20 | O(N²) |
| Row-update per swap | O(N) | M/10 × K | O(N³) dense |
| RR update (k=8) | O(64N) | M/10 × K | O(N²) |
| **Eigensolve per swap** | **O(N³)** | M/10 × K | **O(N⁵)** |

Training at n≤16: N⁵ = 16⁵ ≈ 1M. Fine.

**Inference (any N):**

| Component | Per-call | Calls/episode | Total |
|-----------|----------|---------------|-------|
| Lanczos (warm, k=15) | O(N²) | K=20 | O(N²) |
| Batch-score MLP | O(N²) | K=20 | O(N²) |
| Row-update per swap | O(N) | M/10 × K | O(N³) dense |
| RR update (k=8) | O(64N) | M/10 × K | O(N²) |
| Bridge detection | O(N+M) | M/10 × K | O(N³) |

**No eigensolves at inference.** Total: O(N³) for dense, matching FV/ER.

---

## Directory Structure

```
rl/
├── __init__.py
├── setup.py              # Cython build script
├── current.md            # Current approach documentation
├── train_refine.py       # REFINE v6 training (Edge-MLP + RR + PPO)
├── eval_refine.py        # Evaluate REFINE checkpoints vs baselines
├── train_gnm.py          # Legacy training script (GNN + PPO)
├── eval_checkpoint.py    # Legacy evaluation
├── envs/
│   ├── __init__.py
│   ├── gnm_env.py        # G(n,m) base environment + graph builders
│   └── refine_env.py     # REFINE env with RR tracking
├── models/
│   ├── __init__.py
│   ├── gnm_policy.py     # Legacy GNN policy
│   └── refine_policy.py  # Edge-level MLP policy (~14K params)
├── utils/
│   ├── __init__.py
│   ├── _fast_features.pyx # Cython-accelerated features
│   ├── graph6.py          # Graph6 encoding
│   ├── ours.py            # OURS baseline
│   └── spectral.py        # Lanczos, RR update, eigenvalue utils
└── PoC/                   # Proof-of-concept experiments
    ├── perturbation_poc.py
    ├── rayleigh_ritz_poc.py
    └── rayleigh_ritz_kd_poc.py

src/                       # C implementations (baselines)
├── common.c/h
├── ours.c/h
├── er.c/h
├── fv.c/h
└── main.c
```

---

## Training

```bash
cd rl
python setup.py build_ext --inplace  # Build Cython (first time / after .pyx changes)
python train_refine.py --min-n 8 --max-n 16 --episodes 20000 --delta 2
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-n` | 8 | Minimum graph size |
| `--max-n` | 16 | Maximum graph size |
| `--episodes` | 20000 | Training episodes |
| `--k-steps` | 20 | Lanczos snapshots per episode |
| `--delta` | 2 | Edges added per node per step |
| `--lr` | 3e-4 | Learning rate |
| `--clip-eps` | 0.2 | PPO clipping epsilon |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--ppo-epochs` | 4 | PPO update epochs per episode |
| `--entropy-coef-start` | 0.1 | Entropy bonus start |
| `--entropy-coef-end` | 0.01 | Entropy bonus end |

### Evaluation

```bash
python eval_refine.py logs/best.pt --n 8 --trials 5
python eval_refine.py logs/best.pt --n 8,10,12,14,16 --trials 10
```

---

## Code Principles

1. **No versioned filenames:** Use `gnm_env.py`, not `gnm_env_v2.py`
2. **No dead code:** Delete unused code immediately
3. **Per-swap eigensolvers in training:** Each swap gets its own exact Δλ₂ reward
4. **Inference is O(N³):** No eigensolvers needed — MLP learned from per-swap rewards. Matches FV/ER complexity.
5. **Cython for speed:** Hot-path feature computation is Cython-accelerated; rebuild after changes
6. **Pure Python fallback:** All Cython functions have pure Python equivalents in `spectral.py`

---

## HPC Setup

Training can run on nvwulf cluster:

```bash
# Setup (one-time)
ssh nvwulf "python3 -m venv ~/rl/.venv"
ssh nvwulf "source ~/rl/.venv/bin/activate && pip install torch numpy scipy networkx tqdm cython"

# Build Cython on cluster
rsync -av rl/ nvwulf:~/rl/
ssh nvwulf "cd ~/rl && source .venv/bin/activate && python setup.py build_ext --inplace"

# Run training
ssh nvwulf "cd ~/rl && sbatch train.slurm"
```

---

## Next Steps

1. **Train** REFINE v8 with per-swap rewards (n=8..16, 20K episodes)
2. **Evaluate** against FV/ER/SW baselines at n=8 through n=36
3. **Scale** to n=64, 128, 256, 512, 1024 — verify generalization
4. **Consider** K=100 for large N inference (PoC confirmed RR_K=8 + K=100 works at n=1024)
