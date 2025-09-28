# Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

This repository studies the following combinatorial design problem: among all simple undirected graphs on a fixed number of vertices and edges, construct one whose algebraic connectivity (the Fiedler value) is as large as possible. The training code (`src/train.py`) uses reinforcement learning to build such graphs edge by edge.

**Problem (Maximal Algebraic Connectivity at fixed (n, m)).**

- **Graphs:** We consider simple, undirected, unweighted graphs $G=(V,E)$ with $\vert V\vert=n$ and $\vert E\vert=m$. No self-loops or multi-edges are allowed.
- **Adjacency and Laplacian:** Let $A\in\{0,1\}^{n\times n}$ be the symmetric adjacency matrix with zero diagonal, and $D=\mathrm{diag}(A\mathbf{1})$ the diagonal degree matrix. The (combinatorial) graph Laplacian is $L(G)=D-A\in\mathbb{R}^{n\times n}$.
- **Spectrum:** Denote the eigenvalues of $L(G)$ in nondecreasing order by $0=\lambda_1(L(G))\leq\lambda_2(L(G))\leq\cdots\leq\lambda_n(L(G))$. The quantity $\lambda_2(L(G))$ is the algebraic connectivity (Fiedler value) of $G$, and $\lambda_2(L(G))>0$ iff $G$ is connected.

## Formal Optimization Statement

Let $\mathcal{G}_{n,m}$ be the set of all simple, undirected graphs with $n$ vertices and $m$ edges. The design problem is

$$
\max_{G\in\mathcal{G}_{n,m}} \lambda_2\big(L(G)\big)
$$

- When $m < n-1$, every feasible $G$ is disconnected, so the maximum is $0$. For meaningful connectivity design we typically assume $n-1 \leq m \leq \binom{n}{2}$.
- This problem is combinatorial and non-convex: the search space size is $\binom{\binom{n}{2}}{m}$.

## Equivalent Matrix Formulation

Let $X\in\{0,1\}^{n\times n}$ be a binary symmetric matrix with zero diagonal encoding the graph (i.e., $X=A$). Impose the edge-budget constraint $\sum_{1\leq i \\lt j\leq n} X_{ij} = m$. Writing $D(X)=\mathrm{diag}(X\mathbf{1})$ and $L(X)=D(X)-X$, the problem becomes

$$
\max_{X\in\{0,1\}^{n\times n}} \lambda_2\big(L(X)\big)
$$

subject to

$$
X = X^\top,\quad \mathrm{diag}(X)=\mathbf{0},\quad \sum_{i \\lt j} X_{ij} = m
$$

Multiple optimal graphs can exist; the objective depends only on the spectrum of $L$, not on vertex labels. Adding edges cannot decrease $\lambda_2$, and the complete graph $K_n$ (achieved at $m=\binom{n}{2}$) has $\lambda_2(K_n)=n$.

## Relation to this Repository

- `src/train.py` implements an RL agent that constructs a graph by adding $m-$(initial) edges to maximize the terminal objective $\lambda_2(L(G))$ (or a margin/ratio versus a baseline heuristic). Spectral quantities are computed from the combinatorial Laplacian $L=D-A$.
- Baselines (e.g., effective-resistance greedy) and spectral utilities are provided to evaluate and guide learning.

## Notation Summary

- $n$: number of vertices; $m$: number of edges.
- $A$: adjacency matrix; $D$: degree matrix; $L=D-A$: Laplacian.
- $\lambda_2(L)$: algebraic connectivity (Fiedler value).

This definition section is self-contained and uses LaTeX compatible with standard Markdown renderers (e.g., GitHub and VS Code with math extensions).

## Method Overview

- Environment: Start from a simple initializer (by default, a path on $n$ vertices; i.e., $n-1$ edges). Each episode adds exactly $m-(n-1)$ edges, producing a terminal graph whose Laplacian $L$ defines the reward via $\lambda_2(L)$ or a margin/ratio vs a fixed baseline.
- Policy: A graph-attention network (GAT) scores a restricted set of candidate edges at each step (see pruning below); a categorical policy selects one to add.
- Baseline for evaluation: A greedy effective-resistance (ER) heuristic that repeatedly adds the single highest-ER non-edge until reaching $m$ edges.

## Top‑K Pruning (ER‑Guided)

To tame the combinatorial action space of all $\binom{n}{2}-\vert E\vert$ non-edges, we use an effective‑resistance based Top‑K pruning rule:

- Effective resistance for a non-edge $(u,v)$ under current graph $G$ with Laplacian pseudoinverse $L^+$ is

$$
R_{\mathrm{eff}}(u,v) = (\mathbf{e}_u-\mathbf{e}_v)^\top L^+ (\mathbf{e}_u-\mathbf{e}_v)
= L^+_{uu} + L^+_{vv} - 2 L^+_{uv}.
$$

- At each step, we compute $R_{\mathrm{eff}}$ for all available non-edges (using an eigen decomposition and a zero‑eigenvalue tolerant pseudoinverse) and keep only the K largest values. This Top‑K set is sorted so that the ER top‑1 action is always present at index 0.
- The ER top‑1 action is used to define a strong greedy baseline and, crucially, as a consistent reference inside training/evaluation while the policy learns to choose among the Top‑K.

This ER‑guided Top‑K pruning sharply reduces the action space while retaining the moves most correlated with immediate spectral improvement. It also yields a transparent, high‑quality baseline for reporting.

## Node Features (Spectral Chart + Degree)

Each node $i$ is embedded using a compact, spectral feature vector designed to be informative yet stable:

- Compute the Laplacian eigenvectors $\varphi_2,\varphi_3$ associated with the two smallest nonzero eigenvalues (Fiedler and the next vector). Normalize by removing mean and dividing by the $\ell_2$ norm.
- Form a complex 2D spectral chart $z_i = \varphi_2(i) + \mathrm{i}\varphi_3(i)$ (with the two real components kept explicitly).
- Let $\deg(i)$ be the current degree and $\deg_\mathrm{max}$ its maximum over nodes; define $\mathrm{deg\_norm}(i) = \deg(i)/\max(1,\deg_\mathrm{max})$.
- The node feature is

$$
x_i = \big[\mathrm{deg\_norm}(i),\Re(z_i), \Im(z_i), \varphi_2(i), \varphi_3(i)\big].
$$

This 5D descriptor mixes local connectivity (degree) with global spectral geometry (Fiedler chart), giving the policy a rotationally stable, informative view of the current graph without hand‑crafted labels.

## Training Loop (High Level)

- Initialize graph (path or empty); precompute spectral features.
- Repeat until edge budget is met:
  - Enumerate all non-edges and compute their effective resistances.
  - Keep Top‑K candidates (ER‑guided pruning) and featurize them for the policy.
  - Sample and add one edge; periodically refresh spectra.
- At episode end, compute $\lambda_2(L)$ (and baseline) for logging and learning.

For details, see `src/train.py`.
