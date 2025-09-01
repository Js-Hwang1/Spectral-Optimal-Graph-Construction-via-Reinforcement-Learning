import argparse
import os
import numpy as np


def laplacian_second_eigenpair(adj: np.ndarray):
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


def generate_graph_fgb(n: int, m: int) -> np.ndarray:
    m = max(m, n - 1)
    m = min(m, n * (n - 1) // 2)
    if m == n * (n - 1) // 2:
        adj = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj
    adj = build_path_graph(n)
    cur_m = int(np.sum(adj) // 2)
    while cur_m < m:
        _, phi = laplacian_second_eigenpair(adj)
        best_s = -1.0
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1.0:
                    continue
                s = float(phi[i] - phi[j]) ** 2
                if s > best_s:
                    best_s = s
                    best = (i, j)
        if best is None:
            break
        u, v = best
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        cur_m += 1
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
    parser = argparse.ArgumentParser(description="Fiedler Greedy Baseline (FGB)")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--m", type=int, default=None, help="Number of edges; if omitted, sweep all m and write to out-dir")
    parser.add_argument("--out-dir", type=str, default="/Users/j/Desktop/data/data_FVG", help="Output directory for sweep writing")
    args = parser.parse_args()

    def write_graph_csv(out_dir: str, n: int, m: int, adj: np.ndarray) -> None:
        os.makedirs(out_dir, exist_ok=True)
        lam2 = laplacian_second_eigenvalue(adj)
        path = os.path.join(out_dir, f"data_{n}_{m}.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{lam2:.6f}\n")
            for i in range(n):
                nbrs = [str(j) for j in range(n) if j != i and adj[i, j] > 0.5]
                f.write(f"{i}: " + ", ".join(nbrs) + "\n")
            f.write("\n\n")

    if args.m is None:
        # Sweep m from n-1 to complete graph, writing each
        n = args.n
        m_min = max(0, n - 1)
        m_max = n * (n - 1) // 2
        # Start from path and grow greedily, writing each step
        adj = generate_graph_fgb(n, m_min)
        write_graph_csv(os.path.abspath(args.out_dir), n, m_min, adj)
        for m in range(m_min + 1, m_max + 1):
            adj = generate_graph_fgb(n, m)
            write_graph_csv(os.path.abspath(args.out_dir), n, m, adj)
    else:
        adj = generate_graph_fgb(args.n, args.m)
        lam2 = laplacian_second_eigenvalue(adj)
        print(f"n={args.n} m={args.m} lambda2={lam2:.6f}")


if __name__ == "__main__":
    main()


