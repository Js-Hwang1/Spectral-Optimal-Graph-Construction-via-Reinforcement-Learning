import argparse
import os
import numpy as np
import time


def laplacian(adj: np.ndarray) -> np.ndarray:
    deg = np.sum(adj, axis=1)
    return np.diag(deg) - adj


def laplacian_second_eigenvalue(adj: np.ndarray) -> float:
    L = laplacian(adj)
    vals = np.linalg.eigvalsh(L)
    vals = np.sort(np.real(vals))
    if len(vals) < 2:
        return 0.0
    return float(max(0.0, vals[1]))


def effective_resistance_matrix(adj: np.ndarray) -> np.ndarray:
    """                                     
    Compute effective resistance for all pairs via Laplacian pseudoinverse.
    R_ij = L^+_{ii} + L^+_{jj} - 2 L^+_{ij}
    """
    L = laplacian(adj).astype(np.float64)
    # Moore-Penrose pseudoinverse
    L_pinv = np.linalg.pinv(L, rcond=1e-8)
    diag = np.diag(L_pinv)
    # R = diag[:,None] + diag[None,:] - 2 L_pinv
    R = diag.reshape(-1, 1) + diag.reshape(1, -1) - 2.0 * L_pinv
    # Numerical guard
    R = np.maximum(R, 0.0)
    return R


def build_path_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def generate_graph_erg(n: int, m: int, reoptimize_every: int = 1) -> np.ndarray:
    """
    Effective-Resistance Greedy (ERG):
    Start from a path (connected), iteratively add the missing edge with largest effective resistance
    under the current graph. Recompute resistances every `reoptimize_every` steps.
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
    R = None
    while cur_m < m:
        if steps_since >= reoptimize_every or R is None:
            # If graph disconnected numerically, add small jitter on path ensures connectivity
            R = effective_resistance_matrix(adj)
            steps_since = 0
        best = None
        best_r = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1.0:
                    continue
                r = float(R[i, j])
                if r > best_r:
                    best_r = r
                    best = (i, j)
        if best is None:
            break
        u, v = best
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        cur_m += 1
        steps_since += 1
    return adj


def write_graph_csv(out_dir: str, n: int, m: int, adj: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    lam2 = laplacian_second_eigenvalue(adj)
    path = os.path.join(out_dir, f"data_{n}_{m}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{lam2:.6f}\n")
        for i in range(n):
            nbrs = [str(j) for j in range(n) if j != i and adj[i, j] > 0.5]
            line = f"{i}: " + ", ".join(nbrs)
            f.write(line + "\n")
        f.write("\n\n")


def sweep_erg_write(n: int, out_dir: str, reoptimize_every: int = 1) -> None:
    """
    Generate ERG sequence from m = n-1 up to m = n*(n-1)/2 and write each graph
    as data_<n>_<m>.csv into out_dir with first line lambda2 followed by adjacency list.
    """
    m_min = max(0, n - 1)
    m_max = n * (n - 1) // 2
    # Start from path graph (m = n-1)
    adj = build_path_graph(n)
    cur_m = int(np.sum(adj) // 2)
    # Write the initial path
    write_graph_csv(out_dir, n, cur_m, adj)
    steps_since = reoptimize_every
    R = None
    while cur_m < m_max:
        if steps_since >= reoptimize_every or R is None:
            R = effective_resistance_matrix(adj)
            steps_since = 0
        best = None
        best_r = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1.0:
                    continue
                r = float(R[i, j])
                if r > best_r:
                    best_r = r
                    best = (i, j)
        if best is None:
            break
        u, v = best
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        cur_m += 1
        steps_since += 1
        write_graph_csv(out_dir, n, cur_m, adj)


def main():
    parser = argparse.ArgumentParser(description="Effective-Resistance Greedy baseline")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--m", type=int, default=None, help="Number of edges; if omitted, sweep all m and write to out-dir")
    parser.add_argument("--reoptimize-every", type=int, default=1, help="Recompute resistances every k steps")
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data_ERG"), help="Output directory for sweep writing")
    args = parser.parse_args()

    start_time = time.time()
    if args.m is None:
        # Sweep and write all graphs
        sweep_erg_write(args.n, out_dir=os.path.abspath(args.out_dir), reoptimize_every=args.reoptimize_every)
    else:
        adj = generate_graph_erg(args.n, args.m, reoptimize_every=args.reoptimize_every)
        lam2 = laplacian_second_eigenvalue(adj)
        print(f"n={args.n} m={args.m} lambda2={lam2:.6f} Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()


