"""
Topology generation and mixing matrix construction for DFL benchmark.

Generates adjacency matrices directly in Python for all topologies,
constructs doubly-stochastic gossip mixing matrices via Metropolis-Hastings.

Supported topologies:
  - ring:     Cycle graph (d=2)
  - torus:    2D torus grid (d=4)
  - expander: Exponential graph (d=ceil(log2(n)))
  - random:   Random d-regular via pairing model
  - ours:     Random init + degree equalization (our algorithm)
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
# Our algorithm: random init (m=nd/2 edges) + degree regularization
# =====================================================================

def random_edge_init(n, d, seed=0):
    """
    Phase 1: Generate exactly m = nd/2 edges uniformly at random.
    No surplus, no deficit.
    """
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.uint8)
    count = 0
    while count < m:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1
    return adj


def regularize_deterministic(adj, d):
    """
    Phase 2: Fully deterministic degree equalization.
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


def build_ours(n, d, seed=0):
    """
    Build our adjacency matrix.
    Phase 1: random edge init (m=nd/2 edges), Phase 2: degree equalization.
    Returns n x n uint8 adjacency matrix.
    """
    adj = random_edge_init(n, d, seed=seed)
    adj = regularize_deterministic(adj, d)
    return adj.astype(np.uint8)


# =====================================================================
# ExpGraph — TIME-VARYING one-peer exponential graph
# (Ying et al., "Exponential Graph is Provably Efficient for
#  Decentralized Deep Training", NeurIPS 2021)
#
# L = ceil(log2(n)) matchings.
# Round k: node i pairs with node i XOR 2^k  (power-of-2 n)
#          or uses cyclic shift i ± 2^k mod n (general n).
# Each round is one-peer gossip: W = (1-alpha)*I + alpha*P.
# =====================================================================

def build_expgraph_tv(n, alpha=0.5):
    """
    Build time-varying ExpGraph topology.

    Returns list of (W, adj) pairs, one per matching round.
    For power-of-2 n: L = log2(n) perfect matchings via XOR.
    For general n: L = ceil(log2(n)) rounds with cyclic shift (degree 2).
    """
    L = math.ceil(math.log2(n)) if n > 1 else 1
    is_pow2 = (n & (n - 1)) == 0

    W_list = []
    adj_list = []

    for k in range(L):
        step = 1 << k
        adj = np.zeros((n, n), dtype=np.uint8)

        if is_pow2:
            # XOR matching: i pairs with i ^ step (involution)
            paired = [False] * n
            for i in range(n):
                if paired[i]:
                    continue
                j = i ^ step
                if j < n and j != i:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    paired[i] = True
                    paired[j] = True
        else:
            # General n: cyclic shift ±step (degree 2 per round)
            for i in range(n):
                j_fwd = (i + step) % n
                j_bwd = (i - step) % n
                if j_fwd != i:
                    adj[i, j_fwd] = 1
                    adj[j_fwd, i] = 1
                if j_bwd != i:
                    adj[i, j_bwd] = 1
                    adj[j_bwd, i] = 1

        # Build mixing matrix: W = (1-alpha)*I + alpha * (A / deg)
        # For matching (deg=1): W_ij = alpha, W_ii = 1-alpha
        # For cyclic (deg=2): W_ij = alpha/2, W_ii = 1-alpha
        degrees = adj.sum(axis=1).astype(float)
        W = np.eye(n, dtype=np.float64)
        for i in range(n):
            d_i = degrees[i]
            if d_i > 0:
                W[i, i] = 1.0 - alpha
                for j in range(n):
                    if adj[i, j]:
                        W[i, j] = alpha / d_i

        W_list.append(W)
        adj_list.append(adj)

    return W_list, adj_list


# =====================================================================
# EquiTopo — TIME-VARYING via round-robin 1-factorization
# (Jin et al., "Communication-Efficient Topologies for Decentralized
#  Learning via Equalized Spectral Contribution", NeurIPS 2022)
#
# Constructs d perfect matchings from a round-robin tournament
# (1-factorization of K_n), selects d evenly spaced, cycles through.
# Each round: one-peer gossip with W = (1-alpha)*I + alpha*P.
# =====================================================================

def build_equitopo(n, d, alpha=0.5):
    """
    Build time-varying EquiTopo topology.

    Uses round-robin 1-factorization of K_n to generate n-1 perfect
    matchings, then selects d evenly-spaced ones.

    Returns list of (W, adj) pairs, one per matching round.
    """
    # Need n even for perfect matchings; if odd, use N = n+1 with virtual node
    N = n if n % 2 == 0 else n + 1
    is_odd = (n % 2 != 0)
    total_matchings = N - 1

    # Generate all N-1 perfect matchings via round-robin tournament
    # Round r: pivot (N-1) pairs with r; for j=1..(N-2)/2:
    #   pair ((r-j) mod (N-1), (r+j) mod (N-1))
    all_matchings = []
    for r in range(total_matchings):
        pairs = []
        # Pivot pair
        a, b = r, N - 1
        if a < n and b < n:  # skip if either is virtual
            pairs.append((a, b))
        # Remaining pairs
        for j in range(1, N // 2):
            a = (r - j) % (N - 1)
            b = (r + j) % (N - 1)
            if a < n and b < n and a != b:
                pairs.append((a, b))
        all_matchings.append(pairs)

    # Select d evenly-spaced matchings
    if d >= total_matchings:
        selected = list(range(total_matchings))
    else:
        selected = [int(i * total_matchings / d) for i in range(d)]

    W_list = []
    adj_list = []

    for sel in selected:
        pairs = all_matchings[sel]
        adj = np.zeros((n, n), dtype=np.uint8)
        for a, b in pairs:
            adj[a, b] = 1
            adj[b, a] = 1

        # Build mixing matrix: W = (1-alpha)*I + alpha*P
        W = np.eye(n, dtype=np.float64)
        for a, b in pairs:
            W[a, a] = 1.0 - alpha
            W[a, b] = alpha
            W[b, b] = 1.0 - alpha
            W[b, a] = alpha

        # Nodes without a partner this round keep W_ii = 1
        W_list.append(W)
        adj_list.append(adj)

    return W_list, adj_list


# =====================================================================
# Base-(k+1) time-varying topology
# Real implementation in base_graph.py (ported from Takezawa et al.)
# =====================================================================
# See base_graph.py for the full implementation.


# =====================================================================
# Topology dispatch
# =====================================================================

TOPOLOGY_NAMES = ["ring", "torus", "expander", "random", "ours", "base", "expgraph", "equitopo"]


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
        d: Degree (for ours, random; k for base)
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
    elif name == "ours":
        adj = build_ours(n, d, seed=seed)
    elif name == "expgraph":
        W_np_list, adj_np_list = build_expgraph_tv(n, alpha=0.5)

        # Convert to torch tensors
        W_list = [torch.from_numpy(W).float() for W in W_np_list]

        # Product of all W matrices (one full cycle)
        W_product = np.eye(n, dtype=np.float64)
        for W in W_np_list:
            W_product = W @ W_product

        # Stacked adjacency for lambda2
        adj_stacked = np.zeros((n, n), dtype=np.uint8)
        for adj in adj_np_list:
            adj_stacked = np.maximum(adj_stacked, adj)

        max_deg = max(int(adj.sum(axis=1).max()) for adj in adj_np_list)

        meta = {
            "lambda2": compute_lambda2(adj_stacked),
            "spectral_gap": compute_spectral_gap(W_product),
            "edges": int(adj_stacked.sum()) // 2,
            "degree": max_deg,
            "num_rounds": len(W_np_list),
        }

        return TimeVaryingTopology(W_list, adj_np_list, meta)

    elif name == "equitopo":
        W_np_list, adj_np_list = build_equitopo(n, d, alpha=0.5)

        # Convert to torch tensors
        W_list = [torch.from_numpy(W).float() for W in W_np_list]

        # Product of all W matrices (one full cycle)
        W_product = np.eye(n, dtype=np.float64)
        for W in W_np_list:
            W_product = W @ W_product

        # Stacked adjacency for lambda2
        adj_stacked = np.zeros((n, n), dtype=np.uint8)
        for adj in adj_np_list:
            adj_stacked = np.maximum(adj_stacked, adj)

        meta = {
            "lambda2": compute_lambda2(adj_stacked),
            "spectral_gap": compute_spectral_gap(W_product),
            "edges": int(adj_stacked.sum()) // 2,
            "degree": d,
            "num_rounds": len(W_np_list),
        }

        return TimeVaryingTopology(W_list, adj_np_list, meta)

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
    avg_d = int(degrees.mean()) if np.all(degrees == degrees[0]) else float(degrees.mean())

    W = metropolis_hastings(adj)
    verify_doubly_stochastic(W)

    meta = {
        "lambda2": compute_lambda2(adj),
        "spectral_gap": compute_spectral_gap(W),
        "edges": actual_edges,
        "degree": avg_d,
    }

    W_tensor = torch.from_numpy(W).float()
    return StaticTopology(W_tensor, adj, meta)


def decompose_to_one_peer(static_topo, n, d, seed=0):
    """
    Decompose a static d-regular topology into d one-peer matchings.

    Each matching becomes one round: node exchanges with exactly 1 neighbor.
    Mixing weight per matching: W_k = (1-alpha)*I + alpha*P_k, alpha=0.5.

    Returns a TimeVaryingTopology with d rounds.
    """
    adj = static_topo.adj
    rng = np.random.RandomState(seed)

    # Find d perfect matchings via randomized greedy + augmenting paths
    has_edge = adj.copy().astype(bool)
    matchings = []

    for c in range(d):
        partner = np.full(n, -1)
        order = rng.permutation(n)
        for i in order:
            if partner[i] >= 0:
                continue
            nbrs = np.where(has_edge[i])[0]
            rng.shuffle(nbrs)
            for j in nbrs:
                if partner[j] < 0:
                    partner[i] = j
                    partner[j] = i
                    break
        # Augmenting paths
        for start in range(n):
            if partner[start] >= 0:
                continue
            parent = [-2] * n
            parent[start] = -1
            queue = [start]
            found = -1
            while queue and found < 0:
                u = queue.pop(0)
                for v in range(n):
                    if parent[v] != -2 or not has_edge[u][v] or partner[u] == v:
                        continue
                    parent[v] = u
                    if partner[v] < 0:
                        found = v
                        break
                    w = partner[v]
                    parent[w] = v
                    queue.append(w)
            if found >= 0:
                v = found
                while v >= 0:
                    u = parent[v]
                    prev = parent[u] if u >= 0 else -1
                    partner[u] = v
                    partner[v] = u
                    v = prev
        # Remove matched edges
        for i in range(n):
            j = partner[i]
            if j >= 0:
                has_edge[i][j] = False
                has_edge[j][i] = False
        matchings.append(partner)

    # Build one W matrix per matching: W = (1-alpha)*I + alpha*P
    alpha = 0.5
    W_list = []
    adj_list = []
    for partner in matchings:
        W = np.eye(n, dtype=np.float64)
        a = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            j = partner[i]
            if j >= 0:
                W[i, i] = 1 - alpha
                W[i, j] = alpha
                a[i, j] = 1
                a[j, i] = 1
        W_list.append(torch.from_numpy(W).float())
        adj_list.append(a)

    # Product of all matching W matrices
    W_product = np.eye(n, dtype=np.float64)
    for W_np in [w.numpy().astype(np.float64) for w in W_list]:
        W_product = W_np @ W_product

    meta = {
        "lambda2": static_topo.meta["lambda2"],
        "spectral_gap": compute_spectral_gap(W_product),
        "edges": static_topo.meta["edges"],
        "degree": d,
        "num_rounds": len(matchings),
        "one_peer": True,
    }

    return TimeVaryingTopology(W_list, adj_list, meta)
