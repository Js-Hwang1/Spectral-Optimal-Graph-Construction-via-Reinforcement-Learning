import argparse
import numpy as np


def build_path_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def generate_graph_random(n: int, m: int, seed: int = 0) -> np.ndarray:
    m = max(m, n - 1)
    m = min(m, n * (n - 1) // 2)
    adj = build_path_graph(n)
    cur_m = int(np.sum(adj) // 2)
    rng = np.random.default_rng(seed)
    # add random edges until m
    while cur_m < m:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        u, v = (i, j) if i < j else (j, i)
        if adj[u, v] == 1.0:
            continue
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
    parser = argparse.ArgumentParser(description="Random baseline (connected path + random edges)")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    adj = generate_graph_random(args.n, args.m, seed=args.seed)
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


