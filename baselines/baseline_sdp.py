import argparse
import math
from typing import List, Tuple
import time

import numpy as np


def get_upper_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def laplacian_second_eigenvalue(adj: np.ndarray) -> float:
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    vals = np.linalg.eigvalsh(L)
    vals = np.sort(np.real(vals))
    if len(vals) < 2:
        return 0.0
    return float(max(0.0, vals[1]))


def build_edge_laplacians(n: int) -> List[np.ndarray]:
    laps: List[np.ndarray] = []
    for i in range(n):
        for j in range(i + 1, n):
            L = np.zeros((n, n), dtype=np.float64)
            L[i, i] += 1.0
            L[j, j] += 1.0
            L[i, j] -= 1.0
            L[j, i] -= 1.0
            laps.append(L)
    return laps


def sdp_weights(n: int, m: int, solver: str = "AUTO", verbose: bool = False) -> np.ndarray:
    try:
        import cvxpy as cp  # type: ignore
    except Exception as e:
        raise RuntimeError("cvxpy is required for the SDP baseline. Install via `pip install cvxpy`. ") from e

    pairs = get_upper_pairs(n)
    E = len(pairs)
    L_e = build_edge_laplacians(n)

    w = cp.Variable(E)  # continuous edge weights
    t = cp.Variable()   # lower bound for lambda2 surrogate

    # Centering projector: I - 11^T/n
    I = np.eye(n)
    J = np.ones((n, n)) / float(n)
    P = I - J

    # Laplacian L(w) = sum_e w_e L_e
    Lw = 0
    for k in range(E):
        Lw = Lw + w[k] * L_e[k]

    constraints = [
        Lw - t * P >> 0,
        w >= 0,
        w <= 1,
        cp.sum(w) == float(m),
    ]
    prob = cp.Problem(cp.Maximize(t), constraints)
    # Pick solver
    chosen_solver = None
    if solver == "AUTO":
        if hasattr(cp, "MOSEK") and ("MOSEK" in cp.installed_solvers()):
            chosen_solver = cp.MOSEK
        elif "SCS" in cp.installed_solvers():
            chosen_solver = cp.SCS
        else:
            chosen_solver = None
    else:
        if solver.upper() == "MOSEK" and hasattr(cp, "MOSEK"):
            chosen_solver = cp.MOSEK
        elif solver.upper() == "SCS":
            chosen_solver = cp.SCS

    solver_name = "AUTO"
    if chosen_solver == cp.SCS:
        solver_name = "SCS"
    elif chosen_solver == cp.MOSEK:
        solver_name = "MOSEK"
    print(f"[baseline_sdp] Using solver: {solver_name}")

    if chosen_solver is None:
        # Let cvxpy choose best available
        prob.solve(verbose=verbose)
    elif chosen_solver == cp.SCS:
        prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-6, max_iters=20000)
    elif chosen_solver == cp.MOSEK:
        # If MOSEK license is properly set (MOSEKLM_LICENSE_FILE), this yields tight solutions
        prob.solve(solver=cp.MOSEK, verbose=verbose)
    if w.value is None:
        raise RuntimeError("SDP did not converge to a solution.")
    return np.asarray(w.value, dtype=np.float64)


def round_weights_to_graph(n: int, m: int, w: np.ndarray, trials: int = 32, seed: int = 0, refine_swaps: int = 10, enforce_connectivity: bool = True) -> np.ndarray:
    # Base adjacency starts with a path to ensure connectivity
    if enforce_connectivity:
        adj_base = np.zeros((n, n), dtype=np.float32)
        for i in range(n - 1):
            adj_base[i, i + 1] = 1.0
            adj_base[i + 1, i] = 1.0
        base_edges = n - 1
    else:
        adj_base = np.zeros((n, n), dtype=np.float32)
        base_edges = 0

    need = int(max(0, min(m, n * (n - 1) // 2) - base_edges))
    pairs = get_upper_pairs(n)
    # Exclude edges already in base
    base_set = set()
    if enforce_connectivity:
        for i in range(n - 1):
            base_set.add((i, i + 1))

    cand_idx = [k for k, (i, j) in enumerate(pairs) if (i, j) not in base_set]
    w_cand = w[cand_idx]
    rng = np.random.default_rng(seed)

    def build_adj_from_indices(sel_idx: List[int]) -> np.ndarray:
        adj = adj_base.copy()
        for k in sel_idx:
            i, j = pairs[cand_idx[k]]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return adj

    # Best-of randomized rounding with Gumbel noise + refinement
    best_adj = None
    best_lam = -1.0
    for _ in range(max(1, trials)):
        noise = rng.gumbel(size=w_cand.shape)
        score = w_cand + 0.1 * noise
        order = np.argsort(-score)
        chosen_local = order[:need]
        adj = build_adj_from_indices(list(chosen_local))
        # Local swap refinement
        adj = local_swap_refine(adj, max_swaps=refine_swaps)
        lam = laplacian_second_eigenvalue(adj)
        if lam > best_lam:
            best_lam = lam
            best_adj = adj
    assert best_adj is not None
    return best_adj


def local_swap_refine(adj: np.ndarray, max_swaps: int = 10) -> np.ndarray:
    n = adj.shape[0]
    for _ in range(max_swaps):
        improved = False
        present = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1.0]
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0.0]
        lam0 = laplacian_second_eigenvalue(adj)
        best_adj = adj
        best_lam = lam0
        rng = np.random.default_rng()
        rng.shuffle(present)
        rng.shuffle(missing)
        for (u, v) in present[:min(len(present), 40)]:
            for (a, b) in missing[:min(len(missing), 40)]:
                tmp = adj.copy()
                tmp[u, v] = tmp[v, u] = 0.0
                tmp[a, b] = tmp[b, a] = 1.0
                lam = laplacian_second_eigenvalue(tmp)
                if lam > best_lam + 1e-9:
                    best_lam = lam
                    best_adj = tmp
                    improved = True
        adj = best_adj
        if not improved:
            break
    return adj


def generate_graph_sdp(n: int, m: int, solver: str = "AUTO", verbose: bool = False, trials: int = 32, seed: int = 0, refine_swaps: int = 10) -> np.ndarray:
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    w = sdp_weights(n, m, solver=solver, verbose=verbose)
    adj = round_weights_to_graph(n, m, w, trials=trials, seed=seed, refine_swaps=refine_swaps, enforce_connectivity=True)
    return adj


def main():
    parser = argparse.ArgumentParser(description="SDP baseline for maximizing algebraic connectivity")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--solver", type=str, default="MOSEK", help="AUTO, MOSEK, or SCS")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--refine-swaps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    adj = generate_graph_sdp(args.n, args.m, solver=args.solver, verbose=args.verbose, trials=args.trials, seed=args.seed, refine_swaps=args.refine_swaps)
    lam2 = laplacian_second_eigenvalue(adj)
    print(f"n={args.n} m={args.m} lambda2={lam2:.6f} Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()


