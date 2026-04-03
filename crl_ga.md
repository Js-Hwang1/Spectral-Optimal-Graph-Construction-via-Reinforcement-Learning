# CRL-GA: Genetic Algorithm for Algebraic Connectivity Maximization

## Problem

Given (n, m), construct a graph G with n nodes and m edges that maximizes **algebraic connectivity λ₂** — the second-smallest eigenvalue of the graph Laplacian.

## Algorithm Overview

A population of P graphs evolves over G generations. Each generation: evaluate fitness, select parents via tournament, create offspring via FV-guided crossover, apply FV-guided mutations. Elitism preserves the best individuals.

**Key insight:** Crossover (union of two parent graphs + spectral pruning) discovers structural patterns that no single-trajectory optimization finds. Two mediocre graphs combined can produce an offspring better than either parent.

## Detailed Algorithm

```
INPUT: n, m, pop_size P, generations G, mutation_rate, num_mutations, elite_frac
OUTPUT: best graph found (maximizing λ₂)

1. INITIALIZE population of P random graphs
   For each individual p = 0..P-1:
     adj[p] = ring(n) + random_fill(m - n)    // connected ring + random edges

2. For generation g = 0..G-1:

   2a. EVALUATE FITNESS
       For each individual p:
         fitness[p] = Lanczos_λ₂(adj[p])      // O(N²) warm-start Lanczos

   2b. ELITISM
       Sort by fitness descending
       Copy top elite_frac * P individuals to next generation unchanged

   2c. FILL REMAINING via crossover + mutation
       For each empty slot i in next generation:

         TOURNAMENT SELECT two parents:
           p1 = best of 3 random individuals (by fitness)
           p2 = best of 3 random individuals (by fitness)

         CROSSOVER(p1, p2) → child:
           (i)   Union: child = adj[p1] ∪ adj[p2]         // O(N²)
                 union has ~1.5-2× m edges (parents share some)

           (ii)  Lanczos on union → v₂ (Fiedler vector)   // O(N²)

           (iii) Score all union edges by FV gap:
                 gap(i,j) = (v₂[i] - v₂[j])²
                 Sort ascending (lowest gap = most expendable)

           (iv)  Bridge-accelerated pruning to m edges:     // O(N²)
                 REPEAT:
                   Compute bridges via Tarjan               // O(N²)
                   Remove lowest-gap NON-BRIDGE edges
                 UNTIL edge_count == m or no progress

         MUTATE(child) with probability mutation_rate:
           For s = 0..num_mutations-1:
             Lanczos → v₂                                  // O(N²)
             ADD: softmax sample non-edge by FV gap (high gap preferred)
             REMOVE: softmax sample non-bridge edge by inverted FV gap (low gap preferred)

3. FINAL EVALUATION
   Exact eigensolve on each individual (one-time O(P × N³))
   Return best λ₂ found across all generations
```

## Why Crossover Works

The Fiedler vector v₂ partitions nodes into two groups (positive/negative components). Edges spanning this partition (high FV gap) are the most valuable for connectivity.

**Single-trajectory methods** (FV-greedy, SA, refine) optimize from ONE initial graph. They can only see the Fiedler landscape of their current graph. Edges that look bad from this perspective might be excellent from a different graph's perspective.

**Crossover** combines edges from TWO different graphs. The union graph has a DIFFERENT Fiedler vector — one that reflects the combined connectivity patterns of both parents. When we prune by this union Fiedler, we keep edges that are structurally important in the COMBINED context. Neither parent alone would have discovered this combination.

This is analogous to how in LP, combining two feasible vertices via convex combination can reveal the optimal direction that neither vertex points toward.

## Complexity

| Operation | Per-call | Calls per generation | Total per generation |
|-----------|----------|---------------------|---------------------|
| Lanczos fitness | O(N²) | P | O(P × N²) |
| Crossover: union | O(N²) | P | O(P × N²) |
| Crossover: Lanczos | O(N²) | P | O(P × N²) |
| Crossover: bridge-prune | O(N²) × ~2 | P | O(P × N²) |
| Mutation: Lanczos | O(N²) | P × num_mut | O(P × num_mut × N²) |
| Mutation: bridges | O(N²) | P × num_mut | O(P × num_mut × N²) |

**Per generation:** O(P × num_mut × N²)
**Total over G generations:** O(G × P × num_mut × N²)

**For O(N³) total:** need G × P × num_mut = O(N).

| n | Budget G×P×num_mut | Example: P=20, G=?, mut=3 |
|---|-------------------|--------------------------|
| 16 | 16 | G = 0.27 → not feasible |
| 64 | 64 | G = 1.07 → barely 1 gen |
| 256 | 256 | G = 4.3 |
| 1024 | 1024 | G = 17 |

At small n, the O(N³) budget is tight. At large n (the target), it's workable.

## Performance

**Results (best-of-3 seeds, bridge-based crossover, Lanczos fitness):**

| Config | n=8 vs FV | n=12 vs FV | n=16 vs FV | n=24 vs FV | n=32 vs FV |
|--------|-----------|------------|------------|------------|------------|
| P=10, G=30, mut=1 | 111.4% | 109.5% | 106.0% | — | — |
| P=20, G=50, mut=2 | 114.4% | 112.0% | 109.2% | 103.4% | 100.7% |
| P=20, G=100, mut=3 | 112.5% | 113.4% | 109.8% | 104.5% | 101.2% |

| Config | n=8 vs Best | n=12 vs Best | n=16 vs Best | n=24 vs Best | n=32 vs Best |
|--------|-------------|--------------|--------------|--------------|--------------|
| P=10, G=30, mut=1 | 108.3% | 107.6% | 103.9% | — | — |
| P=20, G=50, mut=2 | 111.1% | 110.0% | 107.1% | 102.4% | 100.0% |
| P=20, G=100, mut=3 | 109.3% | 111.4% | 107.7% | 103.5% | 100.6% |

**Scaling trend:** Performance degrades with n (114% at n=8 → 101% at n=32) because fixed P×G explores a shrinking fraction of the search space. More budget helps: n=24 goes from 103.4% to 104.5% with G=50→100.

**Comparison with other methods (n=16):**

| Method | vs FV | vs Best | Cost |
|--------|-------|---------|------|
| FV baseline | 100% | — | O(N²) |
| SA (iter_mult=10) | 106.2% | 103.0% | O(N⁵) |
| Refine (no metro) | 96.7% | — | O(N³) |
| Refine (metro) | 98.6% | — | O(N⁴) training |
| **GA P=20,G=50** | **109.2%** | **107.1%** | O(P×G×mut×N²) |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pop P` | 20 | Population size |
| `--gen G` | 50 | Number of generations |
| `--mutations M` | 2 | FV-guided swaps per mutation |
| `--mut-rate R` | 0.8 | Probability of mutating a child |
| `--elite F` | 0.1 | Fraction of population preserved via elitism |
| `--seeds S` | 3 | Independent runs per (n,m), report best |

## Build & Run

```bash
make clean && make

# Default: P=20, G=50, mut=2
bin/crl_ga --n 8,12,16 --seeds 3 --baselines baselines.csv

# Larger search
bin/crl_ga --n 8,12,16,24 --seeds 5 --pop 20 --gen 100 --mutations 3 --baselines baselines.csv

# Fast test
bin/crl_ga --n 8 --seeds 1 --pop 10 --gen 10 --mutations 1 --baselines baselines.csv
```

## Files

- `src/ga_main.c` — GA binary (standalone, uses crl.c for Lanczos/bridges)
- `PoC/ga_refine_poc.py` — Original Python PoC (eigensolve fitness)

## Key Optimizations Applied

1. **Lanczos fitness** replaces eigensolve: O(N³) → O(N²) per evaluation. Zero quality degradation — Lanczos preserves fitness ranking perfectly.

2. **Bridge-accelerated pruning** replaces per-removal BFS: O(N⁴) → O(N²) per crossover. Tarjan's bridge detection computed once (or twice), all non-bridge low-gap edges removed in a single pass. Actually **improves** quality by +1-4% vs BFS approach (more aggressive correct pruning).

## Next Steps

- Integrate RL: use GA-discovered graphs as training signal for an RL policy
- Budget-constrained GA: adapt P and G to fit within O(N³) for given n
- Hybrid: GA for exploration + refine for exploitation within same budget
