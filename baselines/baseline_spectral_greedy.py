import argparse
from typing import List, Tuple

import numpy as np


def laplacian_second_eigenpair(adj: np.ndarray) -> Tuple[float, np.ndarray]:
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    vals, vecs = np.linalg.eigh(L)
    order = np.argsort(np.real(vals))
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])
    if len(vals) < 2:
        return 0.0, np.zeros((adj.shape[0],), dtype=float)
    return float(max(0.0, vals[1])), vecs[:, 1]


def build_path_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def generate_graph_spectral_greedy(n: int, m: int, reoptimize_every: int = 10) -> np.ndarray:
    """
    Spectral greedy variant used widely as a baseline: start from a connected path graph, then
    iteratively add the missing edge that maximizes (phi[i]-phi[j])^2 where phi is the current
    Fiedler vector. Recompute phi every `reoptimize_every` steps for speed.
    """
    m = max(m, n - 1)
    m = min(m, n * (n - 1) // 2)
    if m == n * (n - 1) // 2:
        adj = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj

    adj = build_path_graph(n)
    cur_m = int(np.sum(adj) // 2)
    steps_since = reoptimize_every
    lam2, phi = laplacian_second_eigenpair(adj)
    while cur_m < m:
        if steps_since >= reoptimize_every:
            lam2, phi = laplacian_second_eigenpair(adj)
            steps_since = 0
        best_score = -1.0
        best_edge = None
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1.0:
                    continue
                s = float(phi[i] - phi[j]) ** 2
                if s > best_score:
                    best_score = s
                    best_edge = (i, j)
        if best_edge is None:
            break
        u, v = best_edge
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        cur_m += 1
        steps_since += 1
    return adj


def laplacian_second_eigenvalue(adj: np.ndarray) -> float:
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    vals = np.linalg.eigvalsh(L)
    vals = np.sort(np.real(vals))
    if len(vals) < 2:
        return 0.0
    return float(max(0.0, vals[1]))


def main():
    parser = argparse.ArgumentParser(description="Spectral greedy baseline (Fiedler-based)")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--reoptimize-every", type=int, default=10)
    args = parser.parse_args()

    adj = generate_graph_spectral_greedy(args.n, args.m, reoptimize_every=args.reoptimize_every)
    lam2 = laplacian_second_eigenvalue(adj)
    print(f"n={args.n} m={args.m} lambda2={lam2:.6f}")
    deg = np.sum(adj, axis=1)
    deg_prev = deg[0]
    for i in deg:
        deg_curr = i
        if (deg_curr != deg_prev):
            print("Not regular")
            print(deg)
            exit()
        deg_prev = deg_curr
    print("Regular")


if __name__ == "__main__":
    main()


