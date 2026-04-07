"""
Topology generation and mixing matrix construction for DFL benchmark.

Generates adjacency matrices directly in Python for all topologies,
constructs doubly-stochastic gossip mixing matrices via Metropolis-Hastings.

Supported topologies:
  - ring:     Cycle graph (d=2)
  - torus:    2D torus grid (d=4)
  - expander: Exponential graph (d=ceil(log2(n)))
  - random:   Random d-regular via pairing model
  - qrsdr:    QRS-DR (our algorithm)
  - base:     Base-(k+1) time-varying topology

Reference for Metropolis-Hastings mixing weights:
    Xiao & Boyd, "Fast linear iterations for distributed averaging",
    Systems & Control Letters, 2004.
"""

import math
import numpy as np
import torch
from scipy.linalg import eigvalsh


# =====================================================================
# Mixing matrix and spectral analysis
# =====================================================================

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

    for i in range(n):
        W[i, i] = 1.0 - W[i, :].sum()

    return W


def equal_neighbor_weights(adj):
    """
    Equal-neighbor averaging: W_ij = 1/(deg_i + 1) for neighbors and self.
    Used for Base-(k+1) graphs.
    """
    n = adj.shape[0]
    W = np.zeros((n, n), dtype=np.float64)
    degrees = adj.sum(axis=1).astype(float)
    for i in range(n):
        d_i = degrees[i]
        for j in range(n):
            if adj[i, j]:
                W[i, j] = 1.0 / (d_i + 1.0)
        W[i, i] = 1.0 / (d_i + 1.0)
    return W


def compute_spectral_gap(W):
    """
    Spectral gap gamma = 1 - max(|lambda_2|, |lambda_n|).
    Larger gamma = faster mixing.
    """
    eigs = eigvalsh(W)
    eigs_sorted = np.sort(eigs)[::-1]
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


# =====================================================================
# Ring (d=2)
# =====================================================================

def build_ring(n):
    """Build a ring (cycle) adjacency matrix. Degree = 2."""
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


# =====================================================================
# 2D Torus (d=4)
# =====================================================================

def build_torus(n):
    """
    Build a 2D torus grid adjacency matrix. Degree = 4.

    For perfect square n, uses sqrt(n) x sqrt(n).
    Otherwise finds the closest rectangle r x c with r*c = n and r <= c.
    """
    sqrt_n = int(math.isqrt(n))
    if sqrt_n * sqrt_n == n:
        rows, cols = sqrt_n, sqrt_n
    else:
        # Find closest rectangle r x c = n, r <= c
        best_r, best_c = 1, n
        for r in range(2, int(math.isqrt(n)) + 1):
            if n % r == 0:
                c = n // r
                if c >= r:
                    best_r, best_c = r, c
        rows, cols = best_r, best_c

    adj = np.zeros((n, n), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            # Right neighbor (wrap)
            j = r * cols + (c + 1) % cols
            adj[i, j] = 1
            adj[j, i] = 1
            # Down neighbor (wrap)
            j = ((r + 1) % rows) * cols + c
            adj[i, j] = 1
            adj[j, i] = 1

    return adj


# =====================================================================
# Expander / Exponential graph (d = ceil(log2(n)))
# =====================================================================

def build_expander(n):
    """
    Exponential graph. Node i connects to i +/- 2^k (mod n)
    for k = 0, 1, ..., ceil(log2(n)) - 1.

    Degree = 2 * ceil(log2(n)) but may be less due to collisions when n
    is a power of 2.
    """
    k_max = math.ceil(math.log2(n)) if n > 1 else 1
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for k in range(k_max):
            step = (1 << k) % n
            if step == 0:
                continue
            j_plus = (i + step) % n
            j_minus = (i - step) % n
            if j_plus != i:
                adj[i, j_plus] = 1
                adj[j_plus, i] = 1
            if j_minus != i:
                adj[i, j_minus] = 1
                adj[j_minus, i] = 1
    return adj


# =====================================================================
# Random d-regular (pairing model)
# =====================================================================

def build_random_regular(n, d, seed=0):
    """
    Random d-regular graph via the pairing (configuration) model.

    Retries with different random states until a simple graph is produced
    (no self-loops or multi-edges). Uses large seed offsets to decorrelate
    attempts.
    """
    if (n * d) % 2 != 0:
        raise ValueError(f"n*d must be even, got n={n}, d={d}")

    max_attempts = 1000
    for attempt in range(max_attempts):
        rng = np.random.RandomState(seed * 1000 + attempt)
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        rng.shuffle(stubs)

        adj = np.zeros((n, n), dtype=np.uint8)
        ok = True
        for idx in range(0, len(stubs), 2):
            u, v = stubs[idx], stubs[idx + 1]
            if u == v or adj[u, v]:
                ok = False
                break
            adj[u, v] = 1
            adj[v, u] = 1

        if ok:
            return adj

    raise RuntimeError(
        f"Failed to generate simple {d}-regular graph on {n} nodes "
        f"after {max_attempts} attempts (seed={seed})")


# =====================================================================
# QRS-DR (our algorithm) — ported from PoC/qrs_deterministic.py
# =====================================================================

def next_prime(n):
    """Smallest prime > n."""
    if n <= 1:
        return 2
    p = n if n % 2 == 1 else n + 1
    while True:
        if all(p % f != 0 for f in range(2, int(p**0.5) + 1)):
            return p
        p += 2


def primitive_root(p):
    """Smallest primitive root modulo p."""
    if p == 2:
        return 1
    pm1 = p - 1
    factors = set()
    tmp = pm1
    d = 2
    while d * d <= tmp:
        while tmp % d == 0:
            factors.add(d)
            tmp //= d
        d += 1
    if tmp > 1:
        factors.add(tmp)
    for g in range(2, p):
        if all(pow(g, pm1 // f, p) != 1 for f in factors):
            return g
    return 2


def qr_scatter(n, d):
    """QRS-DR Phase 1: QR scatter."""
    adj = np.zeros((n, n), dtype=np.uint8)
    p = next_prime(n + 1)
    g = primitive_root(p)

    for k in range(d):
        c = pow(g, k + 1, p)
        for i in range(n):
            t = (i + 1) % p
            if t == 0:
                t = 1
            j = (t * (t + c)) % p % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


def regularize_deterministic(adj, d):
    """
    QRS-DR Phase 2: Fully deterministic degree regularization.
    All tie-breaking uses LOWEST NODE INDEX.
    """
    n = adj.shape[0]
    adj = adj.copy()
    np.fill_diagonal(adj, 0)
    adj = np.clip(adj, 0, 1)

    target_edges = n * d // 2
    degs = adj.sum(axis=1).astype(int)

    # Step 1: Fix total edge count
    current_edges = int(adj.sum()) // 2

    while current_edges > target_edges:
        max_deg = int(degs.max())
        u = -1
        for i in range(n):
            if degs[i] == max_deg:
                u = i
                break

        neighbors = sorted([v for v in range(n) if adj[u, v] == 1])
        if not neighbors:
            break
        best_deg = max(degs[v] for v in neighbors)
        v = -1
        for nb in neighbors:
            if degs[nb] == best_deg:
                v = nb
                break

        adj[u, v] = 0
        adj[v, u] = 0
        degs[u] -= 1
        degs[v] -= 1
        current_edges -= 1

    while current_edges < target_edges:
        min_deg = int(degs.min())
        u = -1
        for i in range(n):
            if degs[i] == min_deg:
                u = i
                break

        best_v = -1
        best_deg = n + 1
        for v in range(n):
            if v != u and not adj[u, v] and degs[v] < best_deg:
                best_deg = degs[v]
                best_v = v
        if best_v < 0:
            break
        adj[u, best_v] = 1
        adj[best_v, u] = 1
        degs[u] += 1
        degs[best_v] += 1
        current_edges += 1

    # Step 2: Edge-swap to equalize degrees
    for _ in range(n * n):
        over = [i for i in range(n) if degs[i] > d]
        under = [i for i in range(n) if degs[i] < d]
        if not over and not under:
            break
        if not over or not under:
            break

        max_over = max(degs[i] for i in over)
        u = -1
        for i in over:
            if degs[i] == max_over:
                u = i
                break

        min_under = min(degs[i] for i in under)
        w = -1
        for i in under:
            if degs[i] == min_under:
                w = i
                break

        neighbors_u = sorted([v for v in range(n) if adj[u, v] == 1])
        transferred = False
        candidates = sorted(neighbors_u, key=lambda v: (-degs[v], v))
        for v in candidates:
            if v == w:
                continue
            if not adj[w, v]:
                adj[u, v] = 0
                adj[v, u] = 0
                adj[w, v] = 1
                adj[v, w] = 1
                degs[u] -= 1
                degs[w] += 1
                transferred = True
                break

        if transferred:
            continue

        if not adj[u, w]:
            adj[u, w] = 1
            adj[w, u] = 1
            degs[u] += 1
            degs[w] += 1
            neighbors_u = sorted(
                [v for v in range(n) if adj[u, v] == 1 and v != w])
            if neighbors_u:
                best_deg = max(degs[v] for v in neighbors_u)
                v = -1
                for nb in neighbors_u:
                    if degs[nb] == best_deg:
                        v = nb
                        break
                adj[u, v] = 0
                adj[v, u] = 0
                degs[u] -= 1
                degs[v] -= 1
        else:
            best_deg = max(degs[v] for v in neighbors_u)
            v = -1
            for nb in neighbors_u:
                if degs[nb] == best_deg:
                    v = nb
                    break
            adj[u, v] = 0
            adj[v, u] = 0
            degs[u] -= 1
            degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1
                    adj[x, w] = 1
                    degs[w] += 1
                    degs[x] += 1
                    break

    return adj


def build_qrsdr(n, d):
    """
    Build QRS-DR adjacency matrix.
    Phase 1: QR scatter, Phase 2: deterministic regularization.
    Returns n x n uint8 adjacency matrix.
    """
    adj = qr_scatter(n, d)
    adj = regularize_deterministic(adj, d)
    return adj.astype(np.uint8)


# =====================================================================
# Base-(k+1) time-varying topology
# Real implementation in base_graph.py (ported from Takezawa et al.)
# =====================================================================
# See base_graph.py for the full implementation.


# =====================================================================
# Topology dispatch
# =====================================================================

TOPOLOGY_NAMES = ["ring", "torus", "expander", "random", "qrsdr", "base"]


class TimeVaryingTopology:
    """
    Container for time-varying topologies (e.g., Base-(k+1)).
    Holds a list of mixing matrices to cycle through.
    """

    def __init__(self, W_list, adj_list, meta):
        self.W_list = W_list       # list of torch tensors
        self.adj_list = adj_list   # list of numpy arrays
        self.meta = meta
        self.time_varying = True

    def __len__(self):
        return len(self.W_list)


class StaticTopology:
    """Container for static topologies. Single mixing matrix."""

    def __init__(self, W, adj, meta):
        self.W_list = [W]
        self.adj = adj
        self.meta = meta
        self.time_varying = False

    def __len__(self):
        return 1


def get_topology(name, n, d=4, seed=0):
    """
    Build a topology by name and return a topology object.

    Args:
        name: One of TOPOLOGY_NAMES
        n: Number of nodes
        d: Degree (for qrsdr, random; k for base)
        seed: Random seed (for random, base)

    Returns:
        StaticTopology or TimeVaryingTopology with:
            .W_list: list of mixing matrices (torch float tensors)
            .meta: dict with lambda2, spectral_gap, edges, degree
            .time_varying: bool
    """
    if name == "ring":
        adj = build_ring(n)
    elif name == "torus":
        adj = build_torus(n)
    elif name == "expander":
        adj = build_expander(n)
    elif name == "random":
        adj = build_random_regular(n, d, seed=seed)
    elif name == "qrsdr":
        adj = build_qrsdr(n, d)
    elif name == "base":
        # Use the REAL Base-(k+1) implementation (Takezawa et al. NeurIPS 2023)
        from base_graph import build_base_graph as build_real_base
        from base_graph import verify_finite_time_consensus

        W_np_list = build_real_base(n, k=d)

        # Verify finite-time consensus
        ok, err = verify_finite_time_consensus(W_np_list)
        if not ok:
            print(f"WARNING: Base-(k+1) FTC verification failed: err={err:.2e}")

        # Convert to torch tensors
        W_list = [torch.from_numpy(W).float() for W in W_np_list]

        # Build adjacency matrices from W matrices (for metadata)
        adj_list = []
        for W in W_np_list:
            adj = (np.abs(W) > 1e-12).astype(np.uint8)
            np.fill_diagonal(adj, 0)
            adj_list.append(adj)

        # Compute effective mixing: product of all W matrices
        W_product = np.eye(n, dtype=np.float64)
        for W in W_np_list:
            W_product = W @ W_product

        # Max degree across all rounds
        max_deg = max(int((np.abs(W) > 1e-12).sum(axis=1).max()) - 1
                      for W in W_np_list)

        # Stacked adjacency for lambda2
        adj_stacked = np.zeros((n, n), dtype=np.uint8)
        for adj in adj_list:
            adj_stacked = np.maximum(adj_stacked, adj)

        meta = {
            "lambda2": compute_lambda2(adj_stacked),
            "spectral_gap": compute_spectral_gap(W_product),
            "edges": int(adj_stacked.sum()) // 2,
            "degree": max_deg,
            "num_rounds": len(W_np_list),
            "ftc_verified": ok,
        }

        return TimeVaryingTopology(W_list, adj_list, meta)
    else:
        raise ValueError(f"Unknown topology: {name}")

    # Static topology path
    actual_edges = int(adj.sum()) // 2
    degrees = adj.sum(axis=1)

    W = metropolis_hastings(adj)
    verify_doubly_stochastic(W)

    meta = {
        "lambda2": compute_lambda2(adj),
        "spectral_gap": compute_spectral_gap(W),
        "edges": actual_edges,
        "degree": int(degrees.mean()) if np.all(degrees == degrees[0]) else float(degrees.mean()),
    }

    W_tensor = torch.from_numpy(W).float()
    return StaticTopology(W_tensor, adj, meta)
