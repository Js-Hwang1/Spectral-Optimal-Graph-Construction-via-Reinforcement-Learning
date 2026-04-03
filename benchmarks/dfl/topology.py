"""
Topology loading and mixing matrix construction for DFL benchmark.

Loads adjacency matrices from baselines binary format (n*n uint8 row-major),
constructs doubly-stochastic gossip mixing matrices via Metropolis-Hastings.

Reference for Metropolis-Hastings mixing weights:
    Xiao & Boyd, "Fast linear iterations for distributed averaging",
    Systems & Control Letters, 2004.
"""

import os
import numpy as np
import torch
from scipy.linalg import eigvalsh


def load_adj_binary(path, n):
    """Load adjacency matrix from raw n*n uint8 binary file."""
    adj = np.fromfile(path, dtype=np.uint8).reshape(n, n)
    assert np.array_equal(adj, adj.T), f"Adjacency matrix in {path} is not symmetric"
    assert np.all(np.diag(adj) == 0), f"Adjacency matrix in {path} has self-loops"
    return adj


def build_ring(n):
    """Build a ring (cycle) adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def metropolis_hastings(adj):
    """
    Construct Metropolis-Hastings doubly-stochastic mixing matrix.

    W_ij = 1 / (1 + max(d_i, d_j))   for edges (i,j)
    W_ii = 1 - sum_{j != i} W_ij      diagonal
    W_ij = 0                           non-edges

    Guarantees:
      - W is symmetric (since adj is symmetric)
      - W is doubly stochastic (rows and columns sum to 1)
      - All eigenvalues in [-1, 1]
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1).astype(float)
    W = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                w = 1.0 / (1.0 + max(degrees[i], degrees[j]))
                W[i, j] = w
                W[j, i] = w

    # Diagonal: 1 - sum of off-diagonal in row
    for i in range(n):
        W[i, i] = 1.0 - W[i, :].sum()

    return W


def compute_spectral_gap(W):
    """
    Spectral gap gamma = 1 - max(|lambda_2|, |lambda_n|).
    Larger gamma = faster mixing.
    """
    eigs = eigvalsh(W)
    eigs_sorted = np.sort(eigs)[::-1]  # descending
    # eigs_sorted[0] should be 1.0 (doubly stochastic)
    rho = max(abs(eigs_sorted[1]), abs(eigs_sorted[-1]))
    return 1.0 - rho


def compute_lambda2(adj):
    """Algebraic connectivity (lambda_2 of the Laplacian)."""
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def verify_doubly_stochastic(W, tol=1e-10):
    """Verify W is doubly stochastic."""
    row_sums = W.sum(axis=1)
    col_sums = W.sum(axis=0)
    assert np.allclose(row_sums, 1.0, atol=tol), \
        f"Row sums deviate: max |sum-1| = {np.max(np.abs(row_sums - 1))}"
    assert np.allclose(col_sums, 1.0, atol=tol), \
        f"Col sums deviate: max |sum-1| = {np.max(np.abs(col_sums - 1))}"
    assert np.allclose(W, W.T, atol=tol), "W is not symmetric"


def get_topology(name, n, m, seed_id, data_dir="data"):
    """
    Load a topology by name and return (adj, W_tensor, metadata).

    Args:
        name: One of 'ring', 'er', 'fv', 'sw_r25', 'sw_r50', 'sw_r75', 'rd'
        n: Number of nodes
        m: Number of edges
        seed_id: Seed index for stochastic topologies
        data_dir: Directory containing adj binary files

    Returns:
        adj: (n, n) uint8 numpy array
        W: (n, n) float64 torch tensor (mixing matrix)
        meta: dict with lambda2, spectral_gap
    """
    if name == "ring":
        adj = build_ring(n)
    else:
        name_map = {
            "er": f"adj_ER_{n}_{m}_s{seed_id}.bin",
            "fv": f"adj_FV_{n}_{m}_s{seed_id}.bin",
            "sw_r25": f"adj_SW_r25_{n}_{m}_s{seed_id}.bin",
            "sw_r50": f"adj_SW_r50_{n}_{m}_s{seed_id}.bin",
            "sw_r75": f"adj_SW_r75_{n}_{m}_s{seed_id}.bin",
            "rd": f"adj_RD_{n}_{m}_s{seed_id}.bin",
        }
        if name not in name_map:
            raise ValueError(f"Unknown topology: {name}")
        path = os.path.join(data_dir, name_map[name])
        adj = load_adj_binary(path, n)

    actual_m = adj.sum() // 2
    if name != "ring":
        assert actual_m == m, f"Expected m={m}, got {actual_m} in {name}"

    W = metropolis_hastings(adj)
    verify_doubly_stochastic(W)

    meta = {
        "lambda2": compute_lambda2(adj),
        "spectral_gap": compute_spectral_gap(W),
        "edges": int(actual_m),
    }

    W_tensor = torch.from_numpy(W).float()
    return adj, W_tensor, meta


TOPOLOGY_NAMES = ["ring", "sw_r25", "sw_r50", "sw_r75", "er", "fv", "rd"]
