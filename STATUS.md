# CRL Status -- AlphaZero-Style RL for Algebraic Connectivity

**Last updated:** 2026-03-31

---

## Goal

Given `(n, m)`, construct a graph with `n` nodes and `m` edges that maximizes **algebraic connectivity (lambda2)** -- the second-smallest eigenvalue of the graph Laplacian. Train at n=8-10, generalize to n=64-256+. Inference must be **O(N^3)** or better.

---

## Current Best Approach

**AlphaZero-style behavioral cloning** with 3-step exhaustive sweep during training, deployed as greedy MLP at inference. **Warm-started Lanczos** per swap provides spectral features. **Strictly O(N^3) inference.**

### Architecture

- **Shared node embedder:** NE_IN(6) -> H(64) -> H(64), SiLU activations
- **4 scorer heads:**
  - `an`: add node scorer (H->H->1)
  - `at`: add target scorer (TS_IN->H->1, TS_IN=H+2)
  - `rn`: remove node scorer (H->H->1)
  - `rt`: remove target scorer (TS_IN->H->1)

### Node Features (NE_IN = 6)

| # | Feature | Formula |
|---|---------|---------|
| 0 | Degree | `deg[i] / (n-1)` |
| 1 | Step in phase | `step / swaps` |
| 2 | Triangle coefficient | `tri[i] / (deg[i]*(deg[i]-1)/2)` |
| 3 | SND | `snd[i] / (n-1)^2` |
| 4 | Target density | `(m - (n-1)) / (max_m - (n-1))` |
| 5 | **Fiedler component** | `n * v2[i]^2` (from warm Lanczos) |

### Target Scorer Features (TS_IN = H+2)

| # | Feature | Formula |
|---|---------|---------|
| 0-63 | Target node embedding | H-dim from shared embedder |
| 64 | Common neighbors | `cn[src*n+tgt] / n` |
| 65 | **Fiedler gap** | `n * (v2[src] - v2[tgt])^2` |

### Episode Structure (inference)

```
Input: (n, m, seed)
Init: random spanning tree + random edges
      init_tri_snd: O(N^3), init_cn: O(N^3), bridge_init: O(N^2)
Epochs: K = C * N (default C=4)

For each epoch:
    Warm Lanczos -> v2 (once at epoch start)

    ADD phase (3 steps):
        1. node_embed_all with v2 features: O(N)
        2. Score nodes, argmax: O(N)
        3. Score targets with fv_gap: O(deg)
        4. Apply add, update tri/snd/cn/bridges: O(N)
        5. Warm Lanczos refresh: O(N^2)
        (repeat 3 times)

    REM phase (3 steps):
        Same as ADD but for removal (bridge check O(1) lookup)
        Warm Lanczos refresh after each removal: O(N^2)
        (repeat 3 times)

Final: one exact_lambda2 eigensolve: O(N^3)
Return: final lambda2
```

### Inference Complexity -- O(N^3)

| Component | Per swap | Swaps/epoch | Epochs | Total |
|-----------|----------|-------------|--------|-------|
| Warm Lanczos | O(N^2) | 6 | C*N | O(CN^3) |
| Node embed + score | O(N) | 6 | C*N | O(CN^2) |
| Target score | O(N) | 6 | C*N | O(CN^2) |
| tri/snd/cn update | O(N) | 6 | C*N | O(CN^2) |
| Bridge update | O(N) amort | 6 | C*N | O(CN^2) |
| Init (one-time) | | | | O(N^3) |
| Final eigensolve | | | | O(N^3) |
| **TOTAL** | | | | **O(CN^3)** |

---

## Key Findings

### 1. Tabu is Destructive

Tabu was originally designed to force exploration. Our analysis showed:
- Without tabu: agent enters a 3-edge cycle (adds and removes same edges). Lambda2 stays constant.
- With tabu: bans cycled edges, forcing random alternatives. ADD head has no signal, picks random edges. Tabu cascade eventually starves ALL add targets. Agent loses edges net -> lambda2 crashes (70% of baseline).
- **Root cause**: ADD head's scores are near-uniform (entropy 0.999). Tabu bans the only edges the ADD head "knows", leaving it with nothing.

**Decision: no tabu. The cycle is harmless -- it preserves a decent graph.**

### 2. ADD Head Goes Degenerate, REM Head Retains Signal

PoC analysis (feature_signal_poc.py) showed:
- On optimized graphs, ADD node scores have near-zero signal (entropy ratio 0.999, std 0.01)
- REM retains meaningful signal throughout (cn_ij rho=0.82*** with delta-lambda2)
- The "improvement" in early epochs comes from the ADD head's brief signal window (3-5 epochs) on fresh random graphs

### 3. Spectral Features (Fiedler Vector) are Essential

Without v2 features (topological only: degree, tri, snd, cn):
- Agent cycles within 5-10 epochs, can't distinguish spectrally important edges
- At n=32: 70-80% of baselines

With warm Lanczos v2 per epoch:
- Agent has spectral information at every decision point
- At n=32: 100-102% of baselines

Per-swap Lanczos refresh (v2 updated after every edge change):
- Even fresher spectral signal
- +1.2% at n=32, +0.5% at n=64 over once-per-epoch

### 4. Effective Resistance Outperforms Fiedler for Greedy Decisions

Ablation on n=32 (4N epochs, 3 swaps, greedy, no MLP):

| Method | vs FV | vs Best |
|--------|-------|---------|
| Random | 67.3% | 66.6% |
| FV greedy | 90.6% | 89.2% |
| **ER greedy** | **99.6%** | **98.3%** |

ER captures pairwise connectivity strength. FV only captures one spectral cut.

### 5. Epoch Count Scales with N, Not M

Original design: K = C * (max_m - m) epochs. At n=32 m=40, this gave 456 epochs but the agent cycled at epoch 7. 449 wasted epochs.

New design: K = C * N. With C=4: 128 epochs at n=32, 256 at n=64. The agent's productive window is ~10 epochs regardless of m, so scaling with N (not M) is the right budget.

### 6. Epsilon-Greedy + SA Metropolis

Epsilon injection (random node selection with probability epsilon) breaks cycles. Combined with SA-style Metropolis acceptance (using Lanczos lambda2 change), bad perturbations get rejected while good ones are kept. +1-2% over pure greedy.

---

## Results

### Current Best: `spectral_10k` (10k episodes, FV features, per-swap Lanczos)

Pure greedy, no tabu, no epsilon, C=4:

| n | vs FV | vs Best | Wins |
|---|-------|---------|------|
| 32 | 102.4% | 101.7% | 234/466 |
| 64 | 100.9% | 100.2% | 902/1954 |

### Density Breakdown (n=32)

| Density | vs Best |
|---------|---------|
| 0-10% | 83% |
| 10-20% | 93% |
| 20-30% | 96% |
| 30-40% | 98% |
| 40-50% | 102% |
| 50-60% | 102% |
| 60-70% | 102% |
| 70-80% | 101% |
| 80-90% | 101% |
| 90-100% | 100% |

### Scaling

| n | vs Best (greedy) | Trained on |
|---|-----------------|------------|
| 16 | ~103% | n=8-10 |
| 32 | 101.7% | n=8-10 |
| 64 | 100.2% | n=8-10 |
| 96 | 97.6% | n=8-10 |

---

## Training

### AlphaZero-Style Sweep

During training (n=8-10, GPU-accelerated):
1. Agent plays: MLP picks (node, target) via softmax sampling
2. Sweep evaluates: exhaustive 3-step search, GPU batched eigensolve (cuSOLVER)
3. Policy update: KL divergence between MLP softmax and sweep value distribution

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dim (H) | 64 |
| Swaps per epoch | 3 |
| Epoch count | C * N, C=1.0 (training), C=4.0 (inference) |
| Tabu | **0 (disabled)** |
| Sweep temperature | 0.1 |
| Learning rate | 1e-3 (Adam) |
| Train n range | 8-10 |
| Warm Lanczos iters | 10 |

---

## File Structure

```
CRL/
  src/
    test.c              # AZ-style RL (MAIN WORKING FILE)
    sweep_gpu.cu         # GPU batched eigensolve for training
    sweep_gpu.h          # GPU sweep header
    crl.c                # Core library (Lanczos, bridges, graph ops)
    crl.h                # Public API
  PoC/
    feature_signal_poc.py    # Feature-lambda2 correlation analysis
    entropy_trace.py         # Per-epoch trace of scores/entropy/decisions
    ablation_decisions.py    # Random vs FV vs ER greedy comparison
  data/
    ablation_n32.csv         # Raw ablation data for paper
    FV_*.csv, ER_*.csv, SW_*.csv  # Baselines per n
  Makefile                # macOS build (Accelerate)
  Makefile.gpu            # Linux+CUDA build (OpenBLAS + cuSOLVER)
  slurm/                  # HPC job scripts
  baselines.csv           # Combined baselines for n=8-128
  logs/                   # Training checkpoints + eval CSVs
    spectral_10k/final.bin   # CURRENT BEST CHECKPOINT
  CLAUDE.md               # Agent instructions
  STATUS.md               # This file
```

---

## Build & Run

### macOS (M4)
```bash
make clean && make
bin/crl_test --train --min-n 8 --max-n 10 --episodes 10000 --swaps 3 --epoch-C 1.0 --save-dir logs/test
bin/crl_test --checkpoint logs/test/final.bin --n 32,64 --baselines baselines.csv --swaps 3 --epoch-C 4.0 --tabu 0 --csv results.csv
```

### HPC (nvwulf, GPU training)
```bash
rsync -av --exclude='logs/' --exclude='bin/' . nvwulf:/lustre/nvwulf/scratch/jungshwang/CRL/
sbatch slurm/train_spectral_10k.slurm
```

### Eval CLI Flags
```
--n 32,64          Comma-separated n values
--checkpoint X     Model checkpoint (.bin)
--baselines X      Baselines CSV
--swaps 3          Adds+removes per epoch
--epoch-C 4.0      Epoch multiplier (K = C*N)
--tabu 0           Tabu disabled
--epsilon 0        Epsilon-greedy (0=pure greedy)
--csv X            Output per-config CSV
--trace M          Trace single (n,m) with per-epoch stats
```
