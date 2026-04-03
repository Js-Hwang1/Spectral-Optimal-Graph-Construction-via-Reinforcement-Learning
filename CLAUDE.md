# CLAUDE.md — CRL: C-Accelerated RL for Algebraic Connectivity

## Goal

Given `(n, m)`, learn to construct a graph with `n` nodes and `m` edges that maximizes **algebraic connectivity (λ₂)** — the second-smallest eigenvalue of the graph Laplacian.

**Constraints:**
- **Training:** UNRESTRICTED time complexity, but must train inside n=8–24. Results MUST scale to much larger n (n=256+).
- **Inference:** O(N³) total. K=C×N Lanczos epochs, each O(N²). No eigensolves at inference.
- **Target:** Beat FV and ER baselines.
- **Generalization:** Train at n=8–24, generalize to n=32, 64, 128, 256 smoothly.

---

## Hard Rules

### O(N³) Inference Budget

**Inference must be O(N³) or better**, matching the FV/ER baselines we compete against.

- K = C×N Lanczos epochs: K × O(N²) = O(CN³) ✓
- Per-epoch scoring O(N²) + bridge detection O(N²) + sampling O(1) ✓
- Eigensolve for Metropolis: O(N³) per epoch → O(N⁴) total ✗ (training only)
- **No O(N⁴)+ operations anywhere in inference**

### Platform: Apple M4 Mac Mini

- **Accelerate framework** for LAPACK (`dsyev_`) and BLAS (`cblas_sgemm`). Always link `-framework Accelerate`.
- Compile with `-O3 -mcpu=native` for M4 auto-vectorization.
- Prefer `float` for MLP weights, features, logits. Keep `double` for eigensolve numerics.

### Performance-First C

- **Stack-allocate hot-path buffers** when size is bounded (N_MAX).
- **Use `cblas_sgemm` for batch MLP.** (N², hidden_dim) matmuls, not scalar loops.
- **`memcpy`/`memset`** over element-wise loops for bulk init.
- **Profile before micro-optimizing.** Use Instruments (Time Profiler) to find real bottlenecks.

---

## Current Algorithm

### Architecture: Fixed-K Dual-Phase Refine with Metropolis (Full C)

1. **Initialize** graph with ring + random edges
2. **For each epoch** (K = C×N total, C=20 default):
   - Lanczos → v₂ (Fiedler vector, warm-started)
   - Score non-edges by FV gap + MLP correction
   - **Batch-add** B edges (softmax sampling, B = swaps_per_epoch ≈ 1)
   - Lanczos refresh → v₂' of augmented graph
   - Score old edges, detect bridges
   - **Batch-remove** B old edges (softmax sampling)
   - **Metropolis accept/reject** (training: eigensolve; inference: omitted or RL)
3. **Track** best graph seen across all epochs

### Key Design Insight

SA beats FV/ER because of **two mechanisms**: FV-biased proposals + Metropolis acceptance of worsening swaps. This design separates them:
- **FV scoring + Lanczos refresh** handles the proposal quality (cheap, O(N²))
- **Metropolis** handles exploration (expensive, O(N³) eigensolve per epoch)
- **RL's job:** learn a cheap accept/reject oracle to replace Metropolis at inference

### Policy Network (~14.5K params)

- **ADD MLP:** (6) → 64 → 64 → 1 (SiLU) — scores non-edges
- **REM MLP:** (6) → 64 → 64 → 1 (SiLU) — scores edges for removal
- **Value MLP:** (3) → 64 → 64 → 1 (SiLU) — graph-level value estimate
- Scoring: `add_score = fv_gap + MLP_SCALE * mlp(features)`
- MLP_SCALE = 0.1 (dampen MLP relative to FV base)
- Zero-init final layer so initial behavior = pure FV
- Optimizer: AdamW (C implementation)
- Checkpoints: binary `.bin` format

### Edge Features (6-dim)

```
0: n * |v₂ᵢ - v₂ⱼ|²   — Fiedler gap (from Lanczos)
1: degᵢ / (n-1)        — source degree
2: degⱼ / (n-1)        — target degree
3: cycle / K            — progress
4: ref_exists           — does edge exist in reference graph? (0/1)
5: n * |ref_v₂ᵢ - ref_v₂ⱼ|²  — Fiedler gap in reference graph
```

### Graph Features (3-dim)

```
0: mean_degree / (n-1)
1: λ₂ / n
2: cycle / K
```

### Episode Flow

```
Input: (n, m)
Initialize: ring + random edges
total_swaps = m * swap_frac
K = cycle_mult * n
effective_K = min(K, total_swaps)

For epoch = 0..effective_K-1:
  Save graph state (for Metropolis revert)

  Lanczos → v₂                         [O(N²), warm-started]
  Score + sample B adds                 [O(N²) + O(1)]
  Apply adds → G(n, m+B)

  Lanczos → v₂' (refresh)              [O(N²)]
  Bridge detection                      [O(N²)]
  Score + sample B removes              [O(N²) + O(1)]
  Apply removes → G(n, m)

  Metropolis: eigensolve → accept/reject [O(N³), training only]

Output: best graph seen during episode
```

### Complexity

**Training (n=8–24):**

| Component | Per-epoch | Epochs | Total |
|-----------|-----------|--------|-------|
| Lanczos (×2) | O(N²) | K = C×N | O(CN³) |
| Scoring + bridge | O(N²) | K | O(CN³) |
| Eigensolve (Metropolis) | O(N³) | K | O(CN⁴) |

Training at n≤24: C×24⁴ ≈ 7M ops per episode. Acceptable (unrestricted time).

**Inference (any N):**

No eigensolves. Total: **O(CN³)** with C=20. Within budget.

---

## Performance (Current, No RL — Pure FV Scoring)

K=20×N, swap_frac=2.0, geometric cooling τ=0.2→0.002, mean-of-3 trials:

| n | No Metropolis | With Metropolis | Metro Gain |
|---|--------------|----------------|------------|
| 8 | 93.8% of Best | 96.5% | +2.7% |
| 12 | 95.0% | 98.1% | +3.1% |
| 16 | 96.7% | 98.6% | +1.9% |
| 24 | 98.1% | 99.1% | +1.0% |
| 32 | 97.9% | 99.0% | +1.1% |

**Metropolis adds ~2% consistently.** This is the gap RL should fill at inference.

---

## Project Structure

```
CRL/
├── Makefile               # Accelerate on macOS, OpenBLAS on Linux
├── src/
│   ├── crl.h              # Public API: PhaseTxn, MLP, constants, functions
│   ├── crl.c              # Core library: Lanczos, bridges, MLP, dual-phase
│   │                      #   episode collection, PPO, AdamW, checkpoints
│   ├── main.c             # Training binary        → bin/crl_train
│   ├── eval_main.c        # Evaluation binary      → bin/crl_eval
│   ├── ga_main.c          # Genetic algorithm      → bin/crl_ga
│   ├── sa_main.c          # Simulated annealing    → bin/crl_sa
│   ├── v10_main.c         # Node-centric greedy    → bin/crl_v10
│   └── stubs/omp.h        # Single-threaded OMP fallback
├── bin/                   # Build outputs
├── baselines.csv          # Pre-extracted baselines (n,m,fv,er,sw025,sw050,sw075)
├── plans.md               # Development plan and experiment log
├── crl_ga.md              # GA algorithm details and results
├── policy.py              # PyTorch policy (checkpoint conversion only)
├── convert_checkpoint.py  # .bin ↔ .pt conversion
├── PoC/                   # Python proof-of-concept experiments
├── logs/                  # Training checkpoints and logs
└── slurm/                 # HPC job scripts
```

---

## Build & Run

```bash
make clean && make          # builds all into bin/

# Evaluate (no RL, pure FV)
bin/crl_eval logs/zero_mlp.bin --n 16,24,32 --trials 3 --cycle-mult 20 --swap-frac 2.0 --baselines baselines.csv

# Evaluate with Metropolis (expensive, for validation)
bin/crl_eval logs/zero_mlp.bin --n 16,24 --trials 3 --cycle-mult 20 --swap-frac 2.0 --baselines baselines.csv --metropolis

# Train
bin/crl_train --min-n 8 --max-n 16 --episodes 50000 --cycle-mult 20 --swap-frac 0.05

# SA baseline
bin/crl_sa --n 16,24 --seeds 5 --baselines baselines.csv
```

---

## Coding Rules

### 1. Zero Dead Code

- **Delete unused code immediately.** Every line must be live.
- No commented-out blocks, no `#if 0` sections, no "just in case" functions.

### 2. No Versioned Files

- **NEVER create** `v2_main.c`, `crl_v11.c`, `main_old.c`.
- **Update the existing file in place.** Git tracks history.

### 3. PoC Before Mainline

- **Test every new idea in `PoC/` first** using Python.
- Only port to C after the PoC demonstrates the idea works.
- Name PoC files by what they test: `spectral_weights_poc.py`, not `experiment_7.py`.

### 4. Keep This File Current

- **CLAUDE.md is the agent's primary context.** When the algorithm changes, update this file.

### 5. C Style

- `snake_case` everywhere.
- Short functions, single responsibility.
- Comments explain *why*, not *what*.
- Constants in `crl.h` via `#define`. Configuration via `TrainConfig` struct.
- 4-space indent, `{` on same line, K&R style.

### 6. Performance Conventions

- **Hot path = C.** Episode collection, MLP fwd/bwd, eigensolve, GAE, PPO update.
- **Python only for:** checkpoint conversion, verification, PoC experiments.
- **Eigensolves:** Always LAPACK `dsyev_` via Accelerate.
- **MLP batch scoring:** `cblas_sgemm`. Scalar loop only for count < 8.
- **Checkpoints:** Binary `.bin` with `CkptHeader`.

---

## HPC (nvwulf)

```bash
rsync -av --exclude='logs/' --exclude='*.o' . nvwulf:~/CRL/
ssh nvwulf "cd ~/CRL && make clean && make"
ssh nvwulf "cd ~/CRL && sbatch slurm/train.slurm"
```

---

## GA Breakthrough

A genetic algorithm with FV-guided crossover + mutation **massively outperforms SA and all baselines**. C implementation uses Lanczos fitness (O(N²)) + bridge-accelerated crossover pruning (O(N²)).

**Results (C, bridge-based crossover, Lanczos fitness, best-of-3 seeds):**

| Config | n=8 vs Best | n=12 vs Best | n=16 vs Best | n=24 vs Best | n=32 vs Best |
|--------|-------------|--------------|--------------|--------------|--------------|
| P=20, G=50, mut=2 | 111.1% | 110.0% | 107.1% | 102.4% | 100.0% |
| P=20, G=100, mut=3 | 109.3% | 111.4% | 107.7% | 103.5% | 100.6% |

**Comparison at n=16 (vs FV baseline):**
- FV: 100% | SA: 106.2% | Refine+metro: 98.6% | **GA: 109.2%**

**Key optimizations (both lossless):**
1. **Lanczos fitness** replaces eigensolve: O(N³)→O(N²) per eval. Zero quality degradation.
2. **Bridge-accelerated pruning** replaces per-removal BFS: O(N⁴)→O(N²) per crossover. Actually **improved** quality by +1-4%.

**Scaling:** Performance degrades with n (111% at n=8 → 100% at n=32) because fixed P×G explores a shrinking fraction of the search space. More budget (higher G) helps: n=24 goes from 102.4% to 103.5% with G=50→100.

**Complexity:** O(G × P × num_mut × N²). For O(N³): need G×P×num_mut = O(N).

**See `crl_ga.md` for full algorithmic details.**

---

## Current Status

**Algorithm:** Fixed-K dual-phase refine. K=20×N Lanczos epochs, swap_frac=2.0, Metropolis at training time. Achieves 97–99% of best baselines with Metropolis, 94–98% without.

**Next priority:** Investigate GA + RL hybrid to bring GA-level quality (105-108% of baselines) within O(N³) inference budget. See `plans.md` for detailed approach options.

**Baselines:** FV, ER, SW(0.25/0.50/0.75) for n=8–128 in `baselines.csv`.
