"""
Spectral utilities for algebraic connectivity computation.

Provides:
- Exact eigensolvers (for training rewards)
- Normalized Laplacian μ₂ (bounded [0,1] reward signal)
- Fast approximations (for features, not rewards)
"""

import numpy as np
import torch
from typing import Union, Tuple


# =============================================================================
# Exact Eigensolvers (Training Only)
# =============================================================================

def exact_lambda2_np(adj: np.ndarray) -> float:
    """
    Compute exact λ₂ (algebraic connectivity) using numpy.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        λ₂: Second smallest eigenvalue of Laplacian
    """
    n = adj.shape[0]
    if n < 2:
        return 0.0

    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj

    eigvals = np.linalg.eigvalsh(L)

    # λ₂ is second smallest (first is always 0 for connected graphs)
    return float(eigvals[1])


def exact_lambda2_batch(adj_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute exact λ₂ for batch of graphs using torch.

    Args:
        adj_batch: (B, N, N) batch of adjacency matrices

    Returns:
        lambda2: (B,) second smallest eigenvalues
    """
    B, N, _ = adj_batch.shape

    # Build Laplacian: L = D - A
    degrees = adj_batch.sum(dim=-1)  # (B, N)
    L = torch.diag_embed(degrees) - adj_batch  # (B, N, N)

    # Compute all eigenvalues (sorted ascending)
    eigvals = torch.linalg.eigvalsh(L)  # (B, N)

    # Return second smallest
    return eigvals[:, 1]


# =============================================================================
# Normalized Laplacian μ₂ (Primary Reward Signal)
# =============================================================================

def normalized_lambda2_np(adj: np.ndarray) -> float:
    """
    Compute μ₂ of normalized Laplacian using numpy.

    The normalized Laplacian is: L_norm = I - D^{-1/2} A D^{-1/2}
    Its eigenvalues are bounded in [0, 2] for any graph.
    For connected graphs, μ₂ ∈ (0, 1] (approximately).

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        μ₂: Second smallest eigenvalue of normalized Laplacian
             Returns 0 if graph is disconnected.
    """
    n = adj.shape[0]
    if n < 2:
        return 0.0

    degrees = adj.sum(axis=1)

    # Handle isolated nodes (degree 0)
    isolated = degrees == 0
    if isolated.any():
        # Graph is disconnected
        return 0.0

    # D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    # L_norm = I - D^{-1/2} A D^{-1/2}
    # Compute D^{-1/2} A D^{-1/2} efficiently
    normalized_adj = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    L_norm = np.eye(n) - normalized_adj

    eigvals = np.linalg.eigvalsh(L_norm)

    # μ₂ is second smallest
    return float(eigvals[1])


def normalized_lambda2_batch(adj_batch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute μ₂ of normalized Laplacian for batch of graphs.

    Args:
        adj_batch: (B, N, N) batch of adjacency matrices
        eps: Small constant for numerical stability

    Returns:
        mu2: (B,) second smallest eigenvalues of normalized Laplacian
    """
    B, N, _ = adj_batch.shape
    device = adj_batch.device

    # Compute degrees
    degrees = adj_batch.sum(dim=-1)  # (B, N)

    # Handle zero degrees (add eps to avoid division by zero)
    degrees_safe = degrees.clamp(min=eps)

    # D^{-1/2}
    d_inv_sqrt = 1.0 / torch.sqrt(degrees_safe)  # (B, N)

    # L_norm = I - D^{-1/2} A D^{-1/2}
    # Efficient: (B, N, 1) * (B, N, N) * (B, 1, N)
    normalized_adj = adj_batch * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)

    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    L_norm = eye - normalized_adj

    # Compute eigenvalues
    eigvals = torch.linalg.eigvalsh(L_norm)  # (B, N)

    # Return second smallest
    mu2 = eigvals[:, 1]

    # If any graph is disconnected (has zero-degree node), μ₂ should be 0
    has_isolated = (degrees < eps).any(dim=-1)  # (B,)
    mu2 = torch.where(has_isolated, torch.zeros_like(mu2), mu2)

    return mu2


def normalized_lambda2_single(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute μ₂ for single graph (no batch dimension).

    Args:
        adj: (N, N) adjacency matrix
        eps: Small constant for numerical stability

    Returns:
        mu2: Scalar tensor with μ₂ value
    """
    return normalized_lambda2_batch(adj.unsqueeze(0), eps)[0]


# =============================================================================
# Fast Approximations (For Features Only, NOT Rewards)
# =============================================================================

def power_iteration_fiedler(adj: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """
    Approximate Fiedler vector using shifted power iteration.

    Complexity: O(N² × num_iters) = O(N²) for fixed num_iters

    This is for FEATURES, not for reward computation.

    Args:
        adj: (N, N) or (B, N, N) adjacency matrix
        num_iters: Number of power iterations

    Returns:
        fiedler: (N,) or (B, N) approximate Fiedler vector
    """
    if adj.dim() == 2:
        return _power_iteration_fiedler_single(adj, num_iters)
    else:
        return _power_iteration_fiedler_batch(adj, num_iters)


def _power_iteration_fiedler_single(adj: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Power iteration for single graph."""
    n = adj.shape[0]
    device = adj.device
    dtype = adj.dtype

    degrees = adj.sum(dim=-1)

    # Random init orthogonal to all-ones vector
    v = torch.randn(n, device=device, dtype=dtype)
    v = v - v.mean()
    v = v / (v.norm() + 1e-10)

    # Shift for convergence: we want second largest of (shift*I - L)
    # which corresponds to second smallest of L
    max_deg = degrees.max().item()
    shift = 2 * max_deg + 1

    for _ in range(num_iters):
        # v_new = (shift*I - L) @ v = shift*v - D*v + A*v
        v_new = shift * v - degrees * v + adj @ v

        # Re-orthogonalize to constant vector
        v_new = v_new - v_new.mean()

        norm = v_new.norm()
        if norm < 1e-10:
            break
        v = v_new / norm

    return v


def _power_iteration_fiedler_batch(adj: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Power iteration for batch of graphs."""
    B, N, _ = adj.shape
    device = adj.device
    dtype = adj.dtype

    degrees = adj.sum(dim=-1)  # (B, N)

    # Random init
    v = torch.randn(B, N, device=device, dtype=dtype)
    v = v - v.mean(dim=-1, keepdim=True)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-10)

    # Shift per graph
    max_deg = degrees.max(dim=-1)[0]  # (B,)
    shift = 2 * max_deg + 1  # (B,)

    for _ in range(num_iters):
        # v_new = shift*v - D*v + A@v
        v_new = shift.unsqueeze(-1) * v - degrees * v + torch.bmm(adj, v.unsqueeze(-1)).squeeze(-1)

        # Re-orthogonalize
        v_new = v_new - v_new.mean(dim=-1, keepdim=True)

        norm = v_new.norm(dim=-1, keepdim=True)
        v = v_new / (norm + 1e-10)

    return v


def approx_lambda2(adj: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """
    Approximate λ₂ using power iteration and Rayleigh quotient.

    For FEATURES only, not for reward computation.

    Args:
        adj: (N, N) or (B, N, N) adjacency matrix
        num_iters: Number of power iterations

    Returns:
        lambda2: Approximate algebraic connectivity
    """
    fiedler = power_iteration_fiedler(adj, num_iters)

    if adj.dim() == 2:
        degrees = adj.sum(dim=-1)
        Lv = degrees * fiedler - adj @ fiedler
        lambda2 = (fiedler * Lv).sum() / ((fiedler * fiedler).sum() + 1e-10)
    else:
        degrees = adj.sum(dim=-1)
        Lv = degrees * fiedler - torch.bmm(adj, fiedler.unsqueeze(-1)).squeeze(-1)
        lambda2 = (fiedler * Lv).sum(dim=-1) / ((fiedler * fiedler).sum(dim=-1) + 1e-10)

    return lambda2.clamp(min=0)


# =============================================================================
# Connectivity Checking
# =============================================================================

def _bfs_distances(adj: np.ndarray, source: int) -> np.ndarray:
    """
    BFS distance from source to all nodes.

    Args:
        adj: (N, N) adjacency matrix
        source: Source node index

    Returns:
        (N,) integer distance array. Unreachable nodes get distance n.
    """
    n = adj.shape[0]
    dist = np.full(n, n, dtype=np.int32)
    dist[source] = 0
    queue = [source]
    while queue:
        node = queue.pop(0)
        neighbors = np.where(adj[node] > 0)[0]
        for nb in neighbors:
            if dist[nb] == n:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def lanczos_fiedler(adj: np.ndarray, k: int = 20, v_init: np.ndarray = None) -> np.ndarray:
    """
    Approximate Fiedler vector (v₂) via Lanczos iteration.

    Builds k-dim Krylov subspace and extracts best v₂ approximation.
    Same cost per step as power iteration (one mat-vec = O(N²)),
    but converges exponentially faster.

    Args:
        adj: (N, N) adjacency matrix
        k: Number of Lanczos iterations (20 is sufficient for most graphs)
        v_init: Optional warm-start vector (e.g., previous v₂ after a rewire)

    Returns:
        (N,) approximate Fiedler vector, unit norm, orthogonal to all-ones
    """
    n = adj.shape[0]
    if n < 3:
        v = np.array([1.0, -1.0])[:n]
        return v / (np.linalg.norm(v) + 1e-10)

    deg = adj.sum(axis=1)
    k = min(k, n - 1)  # Can't have more Lanczos vectors than n-1

    # Initialize
    if v_init is not None:
        v = v_init.copy()
    else:
        rng = np.random.RandomState(0)
        v = rng.randn(n)

    v = v - v.mean()  # Orthogonal to all-ones (deflate trivial eigenvector)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        rng = np.random.RandomState(0)
        v = rng.randn(n)
        v = v - v.mean()
    v = v / (np.linalg.norm(v) + 1e-10)

    # Lanczos vectors and tridiagonal entries
    V = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    V[:, 0] = v
    actual_k = k

    for j in range(k):
        # w = L @ v_j = D*v_j - A*v_j
        w = deg * V[:, j] - adj @ V[:, j]
        alpha[j] = np.dot(V[:, j], w)

        # Three-term recurrence
        if j > 0:
            w = w - beta[j] * V[:, j - 1]
        w = w - alpha[j] * V[:, j]

        # Full reorthogonalization (prevents loss of orthogonality)
        for i in range(j + 1):
            w = w - np.dot(w, V[:, i]) * V[:, i]

        # Project out all-ones direction
        w = w - w.mean()

        beta_next = np.linalg.norm(w)
        if beta_next < 1e-12:
            actual_k = j + 1
            break

        if j + 1 < k:
            beta[j + 1] = beta_next
            V[:, j + 1] = w / beta_next

    # Solve k×k tridiagonal eigenproblem (cheap)
    T = np.diag(alpha[:actual_k])
    for j in range(actual_k - 1):
        T[j, j + 1] = beta[j + 1]
        T[j + 1, j] = beta[j + 1]

    eig_vals, eig_vecs = np.linalg.eigh(T)

    # Second smallest eigenvalue of T ≈ λ₂ of L
    # Index 0 is smallest (should be ≈ 0 from trivial), index 1 is λ₂
    idx = 1 if actual_k > 1 and eig_vals[0] < 0.1 * max(eig_vals[1], 1e-10) else 0

    # Map back to original space
    v2_approx = V[:, :actual_k] @ eig_vecs[:, idx]
    v2_approx = v2_approx - v2_approx.mean()
    norm = np.linalg.norm(v2_approx)
    if norm > 1e-10:
        v2_approx = v2_approx / norm

    # Sign convention: align with v_init if provided, else make sum positive
    if v_init is not None:
        if np.dot(v2_approx, v_init) < 0:
            v2_approx = -v2_approx
    elif v2_approx.sum() < 0:
        v2_approx = -v2_approx

    return v2_approx


def lanczos_fiedler_ext(adj: np.ndarray, k: int = 20, v_init: np.ndarray = None):
    """
    Extract v₂, v₃, λ₂, λ₃ from a single Lanczos run. Same cost as lanczos_fiedler
    plus one extra O(Nk) projection for v₃.

    Args:
        adj: (N, N) adjacency matrix
        k: Number of Lanczos iterations
        v_init: Optional warm-start vector

    Returns:
        (v2, v3, lam2_ritz, lam3_ritz)
        v2: (N,) approximate Fiedler vector
        v3: (N,) approximate third eigenvector
        lam2_ritz: Ritz value approximating λ₂
        lam3_ritz: Ritz value approximating λ₃
    """
    n = adj.shape[0]
    if n < 4:
        # Too small for meaningful v₃
        v2 = lanczos_fiedler(adj, k=k, v_init=v_init)
        return v2, np.zeros(n), 0.0, 0.0

    deg = adj.sum(axis=1)
    k = min(k, n - 1)

    # Initialize
    if v_init is not None:
        v = v_init.copy()
    else:
        rng = np.random.RandomState(0)
        v = rng.randn(n)

    v = v - v.mean()
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        rng = np.random.RandomState(0)
        v = rng.randn(n)
        v = v - v.mean()
    v = v / (np.linalg.norm(v) + 1e-10)

    # Lanczos vectors and tridiagonal entries
    V = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    V[:, 0] = v
    actual_k = k

    for j in range(k):
        w = deg * V[:, j] - adj @ V[:, j]
        alpha[j] = np.dot(V[:, j], w)

        if j > 0:
            w = w - beta[j] * V[:, j - 1]
        w = w - alpha[j] * V[:, j]

        for i in range(j + 1):
            w = w - np.dot(w, V[:, i]) * V[:, i]

        w = w - w.mean()

        beta_next = np.linalg.norm(w)
        if beta_next < 1e-12:
            actual_k = j + 1
            break

        if j + 1 < k:
            beta[j + 1] = beta_next
            V[:, j + 1] = w / beta_next

    # Solve k×k tridiagonal eigenproblem
    T = np.diag(alpha[:actual_k])
    for j in range(actual_k - 1):
        T[j, j + 1] = beta[j + 1]
        T[j + 1, j] = beta[j + 1]

    eig_vals, eig_vecs = np.linalg.eigh(T)

    # idx for λ₂ (skip trivial eigenvalue ≈ 0)
    idx = 1 if actual_k > 1 and eig_vals[0] < 0.1 * max(eig_vals[1], 1e-10) else 0

    # Extract v₂
    v2_approx = V[:, :actual_k] @ eig_vecs[:, idx]
    v2_approx = v2_approx - v2_approx.mean()
    norm2 = np.linalg.norm(v2_approx)
    if norm2 > 1e-10:
        v2_approx = v2_approx / norm2

    lam2_ritz = float(eig_vals[idx])

    # Extract v₃ (next eigenvector after v₂)
    idx3 = idx + 1
    if idx3 < actual_k:
        v3_approx = V[:, :actual_k] @ eig_vecs[:, idx3]
        v3_approx = v3_approx - v3_approx.mean()
        norm3 = np.linalg.norm(v3_approx)
        if norm3 > 1e-10:
            v3_approx = v3_approx / norm3
        lam3_ritz = float(eig_vals[idx3])
    else:
        v3_approx = np.zeros(n)
        lam3_ritz = lam2_ritz

    # Sign convention for v₂
    if v_init is not None:
        if np.dot(v2_approx, v_init) < 0:
            v2_approx = -v2_approx
    elif v2_approx.sum() < 0:
        v2_approx = -v2_approx

    # Sign convention for v₃: make sum positive (arbitrary but consistent)
    if v3_approx.sum() < 0:
        v3_approx = -v3_approx

    return v2_approx, v3_approx, lam2_ritz, lam3_ritz


def lanczos_fiedler_ext_k(adj: np.ndarray, k: int = 20, v_init: np.ndarray = None,
                          n_eig: int = 8):
    """
    Extract n_eig eigenpairs (v₂..v_{n_eig+1}, λ₂..λ_{n_eig+1}) from a single
    Lanczos run. Same cost as lanczos_fiedler_ext — just projects more Ritz vectors.

    Used to initialize Rayleigh-Ritz subspace tracking.

    Args:
        adj: (N, N) adjacency matrix
        k: Number of Lanczos iterations
        v_init: Optional warm-start vector
        n_eig: Number of eigenpairs to extract (default 8)

    Returns:
        V: (N, n_eig) matrix of eigenvectors [v₂, v₃, ..., v_{n_eig+1}]
        lams: (n_eig,) array of eigenvalues [λ₂, λ₃, ..., λ_{n_eig+1}]
    """
    n = adj.shape[0]
    n_eig = min(n_eig, n - 1)  # Can't have more eigenvectors than n-1

    if n < 3 or n_eig < 1:
        V = np.zeros((n, max(n_eig, 1)))
        if n >= 2:
            V[0, 0] = 1.0 / np.sqrt(2)
            V[1, 0] = -1.0 / np.sqrt(2)
        return V, np.zeros(max(n_eig, 1))

    deg = adj.sum(axis=1)
    k = min(k, n - 1)

    # Initialize
    if v_init is not None:
        v = v_init.copy()
    else:
        rng = np.random.RandomState(0)
        v = rng.randn(n)

    v = v - v.mean()
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        rng = np.random.RandomState(0)
        v = rng.randn(n)
        v = v - v.mean()
    v = v / (np.linalg.norm(v) + 1e-10)

    # Lanczos iteration (identical to lanczos_fiedler_ext)
    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    Q[:, 0] = v
    actual_k = k

    for j in range(k):
        w = deg * Q[:, j] - adj @ Q[:, j]
        alpha[j] = np.dot(Q[:, j], w)

        if j > 0:
            w = w - beta[j] * Q[:, j - 1]
        w = w - alpha[j] * Q[:, j]

        for i in range(j + 1):
            w = w - np.dot(w, Q[:, i]) * Q[:, i]

        w = w - w.mean()

        beta_next = np.linalg.norm(w)
        if beta_next < 1e-12:
            actual_k = j + 1
            break

        if j + 1 < k:
            beta[j + 1] = beta_next
            Q[:, j + 1] = w / beta_next

    # Solve k×k tridiagonal eigenproblem
    T = np.diag(alpha[:actual_k])
    for j in range(actual_k - 1):
        T[j, j + 1] = beta[j + 1]
        T[j + 1, j] = beta[j + 1]

    eig_vals, eig_vecs = np.linalg.eigh(T)

    # idx for λ₂ (skip trivial eigenvalue ≈ 0)
    idx = 1 if actual_k > 1 and eig_vals[0] < 0.1 * max(eig_vals[1], 1e-10) else 0

    # Extract n_eig eigenpairs starting from idx
    n_available = actual_k - idx
    n_extract = min(n_eig, n_available)

    V = np.zeros((n, n_eig))
    lams = np.zeros(n_eig)

    for p in range(n_extract):
        col_idx = idx + p
        vec = Q[:, :actual_k] @ eig_vecs[:, col_idx]
        vec = vec - vec.mean()
        norm_v = np.linalg.norm(vec)
        if norm_v > 1e-10:
            vec = vec / norm_v
        V[:, p] = vec
        lams[p] = float(eig_vals[col_idx])

    # Fill remaining with zeros / repeat last eigenvalue
    for p in range(n_extract, n_eig):
        lams[p] = lams[n_extract - 1] if n_extract > 0 else 0.0

    # Sign convention: v₂ aligned with v_init if provided
    if v_init is not None:
        if np.dot(V[:, 0], v_init) < 0:
            V[:, 0] = -V[:, 0]
    elif V[:, 0].sum() < 0:
        V[:, 0] = -V[:, 0]

    # Sign convention for remaining: make sum positive
    for p in range(1, n_extract):
        if V[:, p].sum() < 0:
            V[:, p] = -V[:, p]

    return V, lams


def rr_update(V: np.ndarray, lams: np.ndarray, u: int, v: int,
              sign: float = 1.0):
    """
    Rayleigh-Ritz k-dimensional subspace update after adding (sign=+1) or
    removing (sign=-1) edge (u, v).

    Projects L_new = L_old + sign * zz^T into span{v₂..v_{k+1}} and solves
    the exact k×k eigenproblem.

    Complexity: O(Nk²) for vector rotation, O(k³) for eigensolver (constant for k≤16).

    Args:
        V: (N, k) matrix of tracked eigenvectors
        lams: (k,) array of tracked eigenvalues
        u, v: edge endpoints
        sign: +1.0 for edge addition, -1.0 for edge removal

    Returns:
        V_new: (N, k) updated eigenvectors
        lams_new: (k,) updated eigenvalues
    """
    # Gap vector: δ_m = V[u,m] - V[v,m]
    delta = V[u, :] - V[v, :]  # (k,)

    # Build k×k projected Laplacian: diag(lams) + sign * δδ^T
    L_sub = np.diag(lams) + sign * np.outer(delta, delta)  # (k, k)

    # Diagonalize — O(k³), constant for k≤16
    new_lams, R = np.linalg.eigh(L_sub)  # sorted ascending

    # Rotate full N-dim vectors: V_new = V @ R  — O(Nk²)
    V_new = V @ R

    return V_new, new_lams


def compute_rwlp_return_prob(adj: np.ndarray) -> np.ndarray:
    """
    Compute P²[i,i] return probability for all nodes. O(N²).

    This is the key spectral proxy for algebraic connectivity:
    - High return prob = node stuck in cluster (bottleneck)
    - Low return prob = node well-mixed (good connectivity)
    Correlates with Fiedler vector components without eigenvectors.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        (N,) array of return probabilities
    """
    degrees = adj.sum(axis=1)
    d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
    P = adj * d_inv[:, None]  # P[i,j] = adj[i,j] / deg[i]
    # diag(P²) via elementwise: O(N²), no matmul
    return (P * P.T).sum(axis=1)


def compute_spd_matrix(adj: np.ndarray) -> np.ndarray:
    """
    All-pairs shortest path distances via BFS.

    O(N × (N + M)) — for sparse graphs O(N²).

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        (N, N) integer SPD matrix
    """
    n = adj.shape[0]
    spd = np.zeros((n, n), dtype=np.int32)
    for source in range(n):
        spd[source] = _bfs_distances(adj, source)
    return spd


def compute_anchor_distances(adj: np.ndarray) -> np.ndarray:
    """
    Compute BFS distances to 3 anchor nodes (normalized).

    Anchors:
        0: node 0
        1: farthest from node 0
        2: farthest from anchor 1

    Returns (N, 3) float array, each column normalized by max distance.
    """
    n = adj.shape[0]

    d0 = _bfs_distances(adj, 0)
    anchor1 = int(np.argmax(d0))
    d1 = _bfs_distances(adj, anchor1)
    anchor2 = int(np.argmax(d1))
    d2 = _bfs_distances(adj, anchor2)

    features = np.zeros((n, 3), dtype=np.float32)
    features[:, 0] = d0 / max(d0.max(), 1)
    features[:, 1] = d1 / max(d1.max(), 1)
    features[:, 2] = d2 / max(d2.max(), 1)

    return features


def is_connected(adj: np.ndarray) -> bool:
    """
    Check if graph is connected using BFS.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        True if connected, False otherwise
    """
    n = adj.shape[0]
    if n == 0:
        return True

    visited = np.zeros(n, dtype=bool)
    queue = [0]
    visited[0] = True
    count = 1

    while queue:
        node = queue.pop(0)
        neighbors = np.where(adj[node] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1

    return count == n


def find_bridges(adj: np.ndarray) -> set:
    """
    Find all bridge edges using iterative Tarjan's algorithm. O(N+M).

    A bridge is an edge whose removal disconnects the graph.
    Iterative (not recursive) to avoid stack overflow at n=1024.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        Set of (min(u,v), max(u,v)) tuples for each bridge edge.
    """
    n = adj.shape[0]
    if n <= 1:
        return set()

    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    bridges = set()
    timer = 0

    # Iterative DFS using explicit stack
    # Stack entries: (node, neighbor_index, phase)
    # phase 0 = first visit (pre-order), phase 1 = returning from child
    for start in range(n):
        if disc[start] != -1:
            continue

        # Build adjacency lists on the fly for this component
        stack = [(start, 0)]  # (node, neighbor_idx)
        disc[start] = low[start] = timer
        timer += 1

        while stack:
            u, ni = stack[-1]
            # Find next unprocessed neighbor
            found_child = False
            neighbors = np.where(adj[u] > 0)[0]

            while ni < len(neighbors):
                v = int(neighbors[ni])
                ni += 1
                stack[-1] = (u, ni)  # Update neighbor index

                if disc[v] == -1:
                    # Tree edge: push child
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, 0))
                    found_child = True
                    break
                elif v != parent[u]:
                    # Back edge: update low
                    low[u] = min(low[u], disc[v])

            if not found_child:
                # Done with all neighbors of u — pop and update parent's low
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    # Check bridge condition: if low[u] > disc[p], edge (p,u) is a bridge
                    if low[u] > disc[p]:
                        bridges.add((min(p, u), max(p, u)))

    return bridges


def is_connected_batch(adj: torch.Tensor) -> torch.Tensor:
    """
    Check connectivity for batch of graphs.

    Uses the fact that a graph is connected iff λ₂ > 0.

    Args:
        adj: (B, N, N) batch of adjacency matrices

    Returns:
        connected: (B,) boolean tensor
    """
    lambda2 = exact_lambda2_batch(adj)
    return lambda2 > 1e-6


# =============================================================================
# Utility Functions
# =============================================================================

def laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute graph Laplacian L = D - A.

    Args:
        adj: (N, N) or (B, N, N) adjacency matrix

    Returns:
        L: Laplacian matrix of same shape
    """
    degrees = adj.sum(dim=-1)
    if adj.dim() == 2:
        return torch.diag(degrees) - adj
    else:
        return torch.diag_embed(degrees) - adj


def normalized_laplacian(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}.

    Args:
        adj: (N, N) or (B, N, N) adjacency matrix
        eps: Small constant for numerical stability

    Returns:
        L_norm: Normalized Laplacian of same shape
    """
    degrees = adj.sum(dim=-1)
    degrees_safe = degrees.clamp(min=eps)
    d_inv_sqrt = 1.0 / torch.sqrt(degrees_safe)

    if adj.dim() == 2:
        normalized_adj = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
        return torch.eye(adj.shape[0], device=adj.device) - normalized_adj
    else:
        normalized_adj = adj * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
        B, N, _ = adj.shape
        eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        return eye - normalized_adj


# =============================================================================
# Cython Acceleration (optional, falls back to pure Python above)
# =============================================================================

try:
    from utils._fast_features import (
        bfs_distances as _bfs_distances,
        compute_anchor_distances,
        is_connected_fast as is_connected,
        find_bridges_fast as find_bridges,
    )
    # Note: compute_rwlp_features not imported from Cython —
    # signature changed to take (adj, u) for SDP-dual features
except ImportError:
    pass
