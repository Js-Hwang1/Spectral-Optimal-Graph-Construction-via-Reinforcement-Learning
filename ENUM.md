# Exact Enumeration of Optimal Lambda2

---

## Overview

For small n, we can enumerate ALL non-isomorphic connected graphs for each (n, m) and find the exact global maximum lambda2. This provides ground truth for training and validation.

---

## Tool: nauty/geng

**geng** (part of nauty by Brendan McKay) generates all non-isomorphic graphs.

```bash
# All connected non-isomorphic graphs with n=8, m=14
geng -c 8 14:14

# Count only (no output)
geng -c -u 8 14:14

# Parallel: split into 4 chunks
geng -c 8 14:14 0/4
geng -c 8 14:14 1/4
geng -c 8 14:14 2/4
geng -c 8 14:14 3/4
```

Output is graph6 format (ASCII, one graph per line).

---

## Non-Isomorphic Graph Counts

| n | Total graphs (all m) | Peak per-m | Time (CPU) |
|---|---------------------|------------|------------|
| 6 | 112 | ~30 | instant |
| 7 | 853 | ~200 | instant |
| 8 | 11,117 | 1,579 | 0.1s |
| 9 | 261,080 | 33,366 | 0.3s |
| 10 | 11,716,571 | 1,348,674 | 8.6s |
| 11 | ~1.0B | ~100M | ~minutes |
| 12 | ~164B | ~15B | ~hours (GPU) |
| 13 | ~50T | ~5T | ~days (GPU) |
| 14 | ~10^16 | -- | months |
| 15 | ~10^18 | -- | centuries |
| 16 | ~10^21 | -- | infeasible |

Growth factor: ~2x per n increase (roughly).

---

## Completed Enumerations

Stored in `data/OPT_{n}.csv` with columns `m,score`.

| n | File | Status |
|---|------|--------|
| 6 | data/OPT_6.csv | Done |
| 7 | data/OPT_7.csv | Done |
| 8 | data/OPT_8.csv | Done |
| 9 | data/OPT_9.csv | Done |
| 10 | data/OPT_10.csv | Done |
| 11 | data/OPT_11.csv | In progress |

---

## Implementation

### CPU (current): `src/enumerate_main.c`

Pipes from geng, parses graph6, computes lambda2 via eigensolve, tracks max per m. OMP parallel across graphs within each m.

```bash
make bin/crl_enum
bin/crl_enum 10        # enumerate n=10
bin/crl_enum 12 30 40  # enumerate n=12, m=30 to m=40 only
```

### GPU (planned)

For n >= 12, the bottleneck is eigensolves. GPU batched eigensolve (cuSOLVER syevjBatched) would accelerate this dramatically.

**Pipeline:**
1. `geng -c n m:m` streams graph6 to stdout
2. CPU parses graph6 into Laplacian matrices, batches them
3. GPU does batched eigensolve
4. CPU reduces to find max lambda2

**No disk needed** -- pipe directly from geng to GPU program.

For parallel multi-GPU: `geng -c n m:m res/mod` splits the generation. Each GPU processes its chunk independently.

---

## Disk Space for Storing Graph6 Files

If pre-generating (not piping):

| n | Bytes per graph | Total graphs | Total size |
|---|----------------|-------------|------------|
| 10 | 9 | 11.7M | ~105 MB |
| 11 | 10 | 1.0B | ~10 GB |
| 12 | 13 | 164B | ~2.1 TB |
| 13 | 15 | 50T | ~755 TB |

**Recommendation:** Don't store. Pipe directly from geng to eigensolve. geng generates faster than any consumer can process.

---

## GPU Throughput Estimates

Eigensolve throughput for small matrices (batched, double precision):

| GPU | FP64 TFLOPS | Est. eigensolves/sec (n=12) | n=12 time | n=13 time |
|-----|------------|----------------------------|-----------|-----------|
| RTX PRO 6000 | 1.5 | ~4M | ~11 hours | ~145 days |
| H200 | 34 | ~90M | ~30 min | ~6.5 days |
| 4x H200 | 136 | ~360M | ~8 min | ~1.6 days |
| 8x H200 (DGX) | 272 | ~720M | ~4 min | ~20 hours |

Key insight: **FP64 throughput is the bottleneck.** Consumer GPUs (RTX) have 1:32 FP64 ratio. Data center GPUs (H200, A100) have 1:2. This gives a ~23x advantage for eigensolve workloads.

---

## Feasibility Summary

| n | CPU (10-core) | 1x RTX PRO 6000 | 1x H200 | 8x H200 |
|---|--------------|-----------------|---------|---------|
| 10 | 9s | <1s | <1s | <1s |
| 11 | minutes | seconds | seconds | seconds |
| 12 | days | 11 hours | 30 min | 4 min |
| 13 | impossible | 145 days | 6.5 days | 20 hours |
| 14+ | impossible | impossible | months+ | weeks+ |

**Conclusion:** Exact enumeration is feasible up to n=12 on a single GPU, n=13 on a multi-GPU node. Beyond n=13, heuristic search (saddle tree, RL) is required.

---

## Baseline Validation

The OPT data reveals baseline bugs. Example:
- n=8, m=27: baselines claim SW(0.75) achieves lambda2=6.8, but enumeration proves the maximum is 6.0 (K8 minus one edge). The baseline is wrong.

OPT data should be used to validate all baseline CSVs and as ground truth for training.
