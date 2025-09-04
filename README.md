# Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

This repository studies the following combinatorial design problem: among all simple undirected graphs on a fixed number of vertices and edges, construct one whose algebraic connectivity (the Fiedler value) is as large as possible. The training code (`src/train.py`) uses reinforcement learning to build such graphs edge by edge.

**Problem (Maximal Algebraic Connectivity at fixed (n, m)).**

- **Graphs:** We consider simple, undirected, unweighted graphs $G=(V,E)$ with $|V|=n$ and $|E|=m$. No self-loops or multi-edges are allowed.
- **Adjacency and Laplacian:** Let $A\in\{0,1\}^{n\times n}$ be the symmetric adjacency matrix with zero diagonal, and $D=\mathrm{diag}(A\mathbf{1})$ the diagonal degree matrix. The (combinatorial) graph Laplacian is $L(G)=D-A\in\mathbb{R}^{n\times n}$.
- **Spectrum:** Denote the eigenvalues of $L(G)$ in nondecreasing order by $0=\lambda_1(L(G))\leq\lambda_2(L(G))\leq\cdots\leq\lambda_n(L(G))$. The quantity $\lambda_2(L(G))$ is the algebraic connectivity (Fiedler value) of $G$, and $\lambda_2(L(G))>0$ iff $G$ is connected.

## Formal Optimization Statement

Let $\mathcal{G}_{n,m}$ be the set of all simple, undirected graphs with $n$ vertices and $m$ edges. The design problem is

$$
\max_{G\in\mathcal{G}_{n,m}}\; \lambda_2\big(L(G)\big).
$$

- When $m < n-1$, every feasible $G$ is disconnected, so the maximum is $0$. For meaningful connectivity design we typically assume $n-1 \leq m \leq \binom{n}{2}$.
- This problem is combinatorial and non-convex: the search space size is $\binom{\binom{n}{2}}{m}$.

## Equivalent Matrix Formulation

Let $X\in\{0,1\}^{n\times n}$ be a binary symmetric matrix with zero diagonal encoding the graph (i.e., $X=A$). Impose the edge-budget constraint $\sum_{1\leq i<j\leq n} X_{ij} = m$. Writing $D(X)=\mathrm{diag}(X\mathbf{1})$ and $L(X)=D(X)-X$, the problem becomes

$$
\max_{X\in\{0,1\}^{n\times n}}\; \lambda_2\big(L(X)\big)
$$

subject to

$$
X = X^\top,\quad \mathrm{diag}(X)=\mathbf{0},\quad \sum_{i<j} X_{ij} = m.
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
