# Decentralized Federated Learning (DFL) Benchmark

## Overview

We evaluate graph topologies constructed by REFINE against baselines (ER, FV, SW, Ring, Random d-regular) by using them as **communication topologies** for decentralized federated learning. The core claim: higher algebraic connectivity (lambda_2) of the communication graph directly translates to faster model convergence, measured in communication rounds.

This benchmark does NOT require a physical cluster. All experiments are simulated on a single GPU — each "node" is a model replica with a private data shard, and communication is restricted to neighbors in the topology graph.

---

## Theoretical Foundation

### Why lambda_2 governs DFL convergence

Given a communication graph G(n,m) with Laplacian L:

1. The gossip mixing matrix is `W = I - L / d_max` (Metropolis-Hastings variant preferred)
2. The spectral gap of W is `gamma = 1 - |1 - lambda_2/d_max|`
3. After T rounds of gossip averaging, the disagreement between nodes decays as `(1 - gamma)^T`
4. Therefore: larger lambda_2 -> larger spectral gap -> faster consensus -> fewer communication rounds to converge

This is not empirical — it is a theorem (Boyd et al., "Fastest Mixing Markov Chain on a Graph", 2004).

### Mixing matrix construction

For a topology G with adjacency matrix A and degree vector d:

**Metropolis-Hastings weights (preferred):**
```
W_ij = 1 / (1 + max(d_i, d_j))      if (i,j) is an edge
W_ii = 1 - sum_{j != i} W_ij         diagonal
W_ij = 0                              otherwise
```

This guarantees W is doubly stochastic (rows and columns sum to 1), which is required for unbiased gossip averaging. The Metropolis-Hastings construction works for any graph topology, including irregular ones.

**Alternative (for d-regular graphs only):**
```
W = I - L / d
```
This is simpler but only valid when all nodes have the same degree.

---

## Experimental Setup

### Model and Dataset

| Parameter     | Value                                          |
|---------------|------------------------------------------------|
| Dataset       | CIFAR-10 (50,000 train / 10,000 test)          |
| Model         | ResNet-20 (0.27M params)                       |
| Optimizer     | SGD, lr=0.1, momentum=0.9, weight_decay=1e-4   |
| LR schedule   | Cosine decay over total communication rounds    |
| Batch size    | 32 per node per local step                      |
| Local steps   | 1 SGD step per communication round (tau=1)      |

ResNet-20 on CIFAR-10 is the standard DFL benchmark model. Reviewers know the expected accuracy (~92% centralized), so deviations are immediately interpretable.

### Graph Sizes and Edge Budgets

| Nodes (n) | Edge budgets m/n (avg degree)       |
|-----------|-------------------------------------|
| 32        | 2, 4, 8, 16                        |
| 64        | 2, 4, 8, 16                        |
| 128       | 2, 4, 8, 16                        |
| 256       | 2, 4, 8, 16                        |

m/n = avg degree. So for n=256, m/n=4 means m=512 edges (each node has ~4 communication links on average).

Sparse regimes (m/n=2,4) are where topology matters most — complete or dense graphs trivially have high lambda_2 but are impractical at scale. The paper's key results should emphasize m/n=4 (realistic for peer-to-peer networks).

### Topologies (Baselines + Ours)

| Topology          | Description                                                | Source             |
|-------------------|------------------------------------------------------------|--------------------|
| Ring              | Cycle graph, lambda_2 ~ O(1/n^2)                          | `baselines/`       |
| SW (rho=0.25)     | Watts-Strogatz, circulant + 25% rewiring                   | `baselines/`       |
| SW (rho=0.50)     | Watts-Strogatz, circulant + 50% rewiring                   | `baselines/`       |
| SW (rho=0.75)     | Watts-Strogatz, circulant + 75% rewiring                   | `baselines/`       |
| ER (greedy)       | Effective resistance greedy edge addition                   | `baselines/`       |
| FV (greedy)       | Fiedler vector greedy edge addition                         | `baselines/`       |
| Random d-regular  | Random d-regular graph (Ramanujan-like)                    | **must add**       |
| REFINE (ours)     | RL-constructed lambda_2-optimal graph                      | `rl/`              |

**Critical: Random d-regular baseline.** Random d-regular graphs have lambda_2 approaching the Ramanujan bound `2*sqrt(d-1)` and are a strong theoretical competitor. If REFINE cannot beat random d-regular graphs, the practical value is limited. This baseline is essential.

### Data Distribution

Run every experiment under **both** data regimes:

#### IID (homogeneous)
Each node receives a uniformly random partition of CIFAR-10. Every node sees all 10 classes. This is the easy case — topology has moderate effect.

#### Non-IID (heterogeneous) — Dirichlet allocation
Use Dirichlet distribution with concentration parameter alpha to control heterogeneity:

| alpha  | Meaning                                                  |
|--------|----------------------------------------------------------|
| 1.0    | Mild heterogeneity — most nodes see most classes         |
| 0.5    | Moderate — nodes have class imbalance                    |
| 0.1    | Severe — each node has 1-2 dominant classes              |

**Implementation:** For each class c, draw a probability vector `p_c ~ Dir(alpha)` over n nodes. Assign each sample of class c to node i with probability `p_c[i]`.

alpha=0.1 is the critical regime. When data is highly non-IID, local models diverge rapidly, and fast gossip mixing (high lambda_2) is essential to pull them back toward consensus. This is where REFINE's advantage should be most visible.

**Report all three alpha values.** alpha=0.1 is the headline result; alpha=1.0 shows the method doesn't hurt in easy settings.

### Communication Rounds and Evaluation

| Parameter               | Value                         |
|-------------------------|-------------------------------|
| Total communication rounds | 2000 (n=32,64), 3000 (n=128,256) |
| Evaluation frequency    | Every 10 rounds               |
| Metric                  | Top-1 test accuracy (global model = average of all node weights) |
| Seeds                   | 5 per configuration           |

The "global model" at evaluation time is the average of all n node models: `w_global = (1/n) * sum(w_i)`. This is the model that would be deployed. Test it on the full CIFAR-10 test set.

---

## Simulation Protocol

### Per-round pseudocode

```
Input: topology G(n,m), mixing matrix W, node data shards D_1..D_n

Initialize: all nodes share the same random init w_0
w_i = w_0 for all i

for round t = 1 to T:
    # 1. Local SGD step
    for each node i:
        sample batch B_i from D_i
        g_i = gradient(loss(w_i, B_i))
        w_i = w_i - lr * g_i

    # 2. Gossip averaging (one round)
    w_new = {}
    for each node i:
        w_new[i] = sum(W[i][j] * w_j for all j)  # includes W[i][i] * w_i
    w = w_new

    # 3. Evaluate (every 10 rounds)
    if t % 10 == 0:
        w_global = mean(w_i for all i)
        acc = evaluate(w_global, test_set)
        log(t, acc)
```

### Implementation notes

- **All node models live on one GPU.** Stack them as a batch of n model replicas. The gossip step is a matrix multiply: `W_params = W @ node_params` (where node_params is [n, num_params]).
- **Memory estimate:** ResNet-20 has 0.27M params (float32 = 1.08 MB per model). For n=256: 256 * 1.08 MB = 277 MB. Trivially fits on any GPU.
- **Gossip as matmul:** Store all node parameters as a 2D tensor [n, P] where P = total params. The gossip step is `params = W @ params`. W is [n, n], precomputed from the topology. This is a single matmul — very fast.
- **Same init across topologies.** For a given seed, ALL topologies must start from the same random initialization. The only variable is W (the mixing matrix derived from the topology).

---

## What to Report

### Primary figures (main paper)

**Figure 1: Test accuracy vs communication rounds**
- One subplot per n (32, 64, 128, 256)
- Fix m/n = 4 (the practical regime)
- alpha = 0.1 (non-IID, where topology matters most)
- Lines: Ring, SW(0.50), ER, FV, Random d-regular, REFINE
- Shaded regions: +/- 1 std over 5 seeds
- Expected result: REFINE converges in fewest rounds; Ring is slowest

**Figure 2: Rounds to reach 85% accuracy vs n**
- X-axis: n (32, 64, 128, 256), log scale
- Y-axis: communication rounds to first reach 85% test accuracy
- One line per topology
- Fix m/n = 4, alpha = 0.1
- Expected result: Ring scales as O(n^2), REFINE scales much better

**Figure 3: lambda_2 vs convergence rate (scatter)**
- Each point is one (topology, n, m/n) configuration
- X-axis: lambda_2 of the topology
- Y-axis: rounds to reach 85% accuracy
- Show the theoretical 1/lambda_2 curve overlaid
- Expected result: points cluster around the theoretical prediction, confirming lambda_2 is the governing quantity

**Table 1: Final accuracy at T=2000 rounds**
- Rows: topologies
- Columns: (n, m/n, alpha) configurations
- Report mean +/- std
- Bold the best per column

### Secondary figures (appendix)

**Figure A1: Effect of edge budget**
- Fix n=128, alpha=0.1
- Subplots for m/n = 2, 4, 8, 16
- Shows that topology matters most when edges are scarce

**Figure A2: Effect of data heterogeneity**
- Fix n=128, m/n=4
- Subplots for alpha = 1.0, 0.5, 0.1
- Shows REFINE's advantage grows as heterogeneity increases

**Figure A3: lambda_2 comparison table**
- Pure algebraic connectivity scores for each (n, m) across all topologies
- Mean +/- std from baselines/
- This connects the DFL results back to the core algorithmic contribution

---

## Total Experiment Count

```
Topologies:   7 (Ring, SW x3, ER, FV, Random d-reg) + REFINE = 8
Node counts:  4 (32, 64, 128, 256)
Edge budgets: 4 (m/n = 2, 4, 8, 16)
Data splits:  3 (alpha = 1.0, 0.5, 0.1)
Seeds:        5

Total: 8 * 4 * 4 * 3 * 5 = 1920 runs
```

At ~10-30 min per run on a single GPU: ~320-960 GPU-hours.
With instant single-GPU scheduling on the Blackwell cluster, this completes in 3-7 days of wall time with 8-16 concurrent jobs.

---

## Filesystem Structure

```
benchmarks/
  dfl/
    train.py            # Main DFL simulation loop
    topology.py         # Load topology from baselines/ CSVs, build mixing matrix
    data.py             # CIFAR-10 partitioning (IID + Dirichlet non-IID)
    models.py           # ResNet-20 definition
    eval.py             # Post-hoc analysis: load logs, generate plots/tables
    slurm/
      run_single.slurm  # Single-GPU job script
      sweep.sh          # Launch all configs
    results/            # Output logs (gitignored)
```

---

## Checklist Before Submission

- [ ] All topologies use the same (n, m) budget — apples-to-apples
- [ ] Mixing matrix W is doubly stochastic (verify row and column sums = 1)
- [ ] Same random init for all topologies within a seed
- [ ] Non-IID splits are identical across topologies within a seed
- [ ] Random d-regular baseline is included (strongest theoretical competitor)
- [ ] Report rounds-to-accuracy, not just final accuracy
- [ ] lambda_2 values reported alongside DFL results (connect the two)
- [ ] Error bars on all plots (5 seeds minimum)
- [ ] Test with tau > 1 (multiple local steps) as an ablation if reviewers ask
