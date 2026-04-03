"""
Greedy Fiedler Rewire baseline.

Starts from random tree + random edges (seeded, reproducible),
then greedily rewires using the Fiedler cut score.

Algorithm per round (φ rounds total):
  For each node u in [0, n):
    1. Compute Fiedler vector v₂ (exact eigensolver)
    2. Cut score for each node i: (v₂[u] - v₂[i])²
    3. Find neighbor v with LOWEST cut score → candidate to drop
    4. Find non-neighbor w with HIGHEST cut score → candidate to connect
    5. If cut_score(w) > cut_score(v): try rewire (u,v) → (u,w)
       - Check connectivity (BFS)
       - Recompute λ₂; accept only if λ₂ improved

Usage:
    from utils.greedy_fiedler import greedy_fiedler_rewire
    adj, lambda2, info = greedy_fiedler_rewire(n=16, m=32, seed=42)
"""

import numpy as np
from utils.spectral import is_connected


def greedy_fiedler_rewire(n: int, m: int, seed: int = 0, verbose: bool = False):
    """
    Greedy Fiedler rewire baseline.

    Args:
        n: Number of nodes
        m: Number of edges
        seed: Random seed for reproducible initial graph
        verbose: Print per-round stats

    Returns:
        adj: Final adjacency matrix (n, n)
        lambda2: Final algebraic connectivity
        info: Dict with stats
    """
    from envs.gnm_env import build_random_tree_initial, compute_phi

    rng = np.random.RandomState(seed)
    adj = build_random_tree_initial(n, m, rng)
    degrees = adj.sum(axis=1)

    # Initial λ₂
    L = np.diag(degrees) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    lambda2 = float(eigvals[1])
    fiedler = eigvecs[:, 1]
    initial_lambda2 = lambda2

    phi = compute_phi(n, m)
    total_rewires = 0
    total_attempts = 0

    for rnd in range(phi):
        round_rewires = 0

        for u in range(n):
            # Cut score for each node relative to u
            cut_scores = (fiedler[u] - fiedler) ** 2

            # Neighbors of u
            neighbors = np.where(adj[u] > 0)[0]
            if len(neighbors) == 0:
                continue

            # Non-neighbors of u (excluding self)
            non_neighbors = np.where((adj[u] == 0) & (np.arange(n) != u))[0]
            if len(non_neighbors) == 0:
                continue

            # Neighbor with lowest cut score → least useful edge
            v = neighbors[np.argmin(cut_scores[neighbors])]

            # Non-neighbor with highest cut score → best destination
            w = non_neighbors[np.argmax(cut_scores[non_neighbors])]

            # Only attempt if destination is better than what we'd drop
            if cut_scores[w] <= cut_scores[v]:
                continue

            total_attempts += 1

            # Try rewire: remove (u,v), add (u,w)
            adj[u, v] = adj[v, u] = 0.0
            adj[u, w] = adj[w, u] = 1.0

            if not is_connected(adj):
                # Revert
                adj[u, v] = adj[v, u] = 1.0
                adj[u, w] = adj[w, u] = 0.0
                continue

            # Recompute λ₂
            degrees[v] -= 1
            degrees[w] += 1
            L = np.diag(degrees) - adj
            eigvals, eigvecs = np.linalg.eigh(L)
            new_lambda2 = float(eigvals[1])

            if new_lambda2 > lambda2:
                # Accept
                lambda2 = new_lambda2
                fiedler = eigvecs[:, 1]
                total_rewires += 1
                round_rewires += 1
            else:
                # Revert
                adj[u, w] = adj[w, u] = 0.0
                adj[u, v] = adj[v, u] = 1.0
                degrees[w] -= 1
                degrees[v] += 1

        if verbose:
            print(f"  Round {rnd+1}/{phi}: {round_rewires} rewires, λ₂={lambda2:.4f}")

    info = {
        'initial_lambda2': initial_lambda2,
        'final_lambda2': lambda2,
        'improvement': lambda2 - initial_lambda2,
        'total_rewires': total_rewires,
        'total_attempts': total_attempts,
        'phi': phi,
        'n': n,
        'm': m,
    }

    return adj, lambda2, info


def eval_greedy_fiedler(n: int, m: int, num_trials: int = 5, verbose: bool = False):
    """
    Evaluate greedy Fiedler rewire over multiple trials.

    Returns:
        mean_lambda2, std_lambda2, results_list
    """
    results = []
    for seed in range(num_trials):
        _, lam2, info = greedy_fiedler_rewire(n, m, seed=seed, verbose=verbose)
        results.append(info)

    lambda2s = [r['final_lambda2'] for r in results]
    return np.mean(lambda2s), np.std(lambda2s), results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Greedy Fiedler Rewire baseline")
    parser.add_argument("--n", type=str, required=True, help="Comma-separated n values")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]

    for n in n_values:
        max_m = n * (n - 1) // 2
        min_m = n - 1
        print(f"\n{'='*70}")
        print(f"n={n}, sweeping m={min_m}..{max_m}")
        print(f"{'='*70}")
        print(f"{'m':>5} {'density':>8} {'phi':>4} {'greedy':>9} {'std':>7} {'rewires':>8}")
        print("-" * 50)

        from envs.gnm_env import compute_phi

        for m in range(min_m, max_m + 1):
            mean_l2, std_l2, results = eval_greedy_fiedler(
                n, m, num_trials=args.trials, verbose=args.verbose
            )
            phi = compute_phi(n, m)
            density = m / max_m
            avg_rw = np.mean([r['total_rewires'] for r in results])
            print(f"{m:>5} {density:>8.3f} {phi:>4} {mean_l2:>9.4f} {std_l2:>7.4f} {avg_rw:>8.1f}")
