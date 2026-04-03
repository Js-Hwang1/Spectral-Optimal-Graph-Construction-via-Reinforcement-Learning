"""
G(n,m) Foundational Model Environment.

Policy decides edge rewiring to maximize algebraic connectivity λ₂.

Input: (n, m) where:
  - n: number of nodes (integer > 2)
  - m: number of edges, n-1 <= m <= n*(n-1)/2

Features: degree + 2-hop neighbor degree sum (no Lanczos/eigensolvers at inference).
R = min(m//4, M, NE) swaps per step — scales with edge count.
No BFS: training detects disconnection via λ₂ < ε (free from eigensolver), inference trusts policy.
"""

import math
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, NamedTuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random

from utils.spectral import is_connected  # kept for batch_step legacy path

NODE_FEAT_DIM = 3    # degree_norm, 2-hop reachability, 3-hop reachability
PAIR_FEAT_DIM = 6    # product + imbalance for each of 3 node features
GRAPH_FEAT_DIM = 3   # density, step_frac, n/max_n
K_STEPS = 20         # Number of rewiring steps per episode
SHAPING_COEF = 1.0   # Dense per-step reward coefficient
DISCONNECT_PENALTY = -10.0  # Heavy penalty for disconnecting the graph


def algebraic_connectivity(adj: np.ndarray) -> float:
    """Compute λ₂ - second smallest eigenvalue of Laplacian."""
    n = adj.shape[0]
    if n < 2:
        return 0.0
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    eigvals = np.linalg.eigvalsh(L)
    return float(eigvals[1])


def compute_node_features(
    adj: np.ndarray,
    degrees: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Compute 3-dim node features. O(N²) via two chained mat-vecs.

    Features per node:
      [0] deg[i] / (n-1)                        — normalized degree (1-hop)
      [1] (adj @ degrees)[i] / (n-1)²           — 2-hop reachability
      [2] (adj @ adj @ degrees)[i] / (n-1)³     — 3-hop reachability

    For a d-regular graph: feat ≈ [d, d², d³], all in [0,1].

    Args:
        adj: (n, n) adjacency matrix
        degrees: (n,) degree array
        n: number of nodes

    Returns:
        (n, 3) node feature matrix
    """
    nm1 = max(n - 1, 1)
    node_feat = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)

    # 1-hop: degree
    node_feat[:, 0] = degrees / nm1

    # 2-hop: adj @ degrees — O(N²) mat-vec
    hop2 = adj @ degrees
    node_feat[:, 1] = hop2 / (nm1 * nm1)

    # 3-hop: adj @ hop2 — O(N²) mat-vec
    hop3 = adj @ hop2
    node_feat[:, 2] = hop3 / (nm1 * nm1 * nm1)

    return node_feat


def compute_pair_features(
    node_feat: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Compute 6-dim pair features from node features. O(P) where P = |indices|.

    Features per pair (i, j) — product + imbalance for each hop:
      [0] feat[0,i] * feat[0,j]              — degree product
      [1] (feat[0,i] - feat[0,j])²           — degree imbalance
      [2] feat[1,i] + feat[1,j]              — 2-hop sum
      [3] (feat[1,i] - feat[1,j])²           — 2-hop imbalance
      [4] feat[2,i] + feat[2,j]              — 3-hop sum
      [5] (feat[2,i] - feat[2,j])²           — 3-hop imbalance

    Args:
        node_feat: (n, 3) node features [deg_norm, 2-hop, 3-hop]
        indices: (P, 2) array of (i, j) pairs

    Returns:
        (P, 6) pair feature matrix
    """
    if len(indices) == 0:
        return np.zeros((0, PAIR_FEAT_DIM), dtype=np.float32)

    i_idx = indices[:, 0]
    j_idx = indices[:, 1]
    fi = node_feat[i_idx]  # (P, 3)
    fj = node_feat[j_idx]  # (P, 3)

    pair_feat = np.zeros((len(indices), PAIR_FEAT_DIM), dtype=np.float32)
    pair_feat[:, 0] = fi[:, 0] * fj[:, 0]           # deg product
    pair_feat[:, 1] = (fi[:, 0] - fj[:, 0]) ** 2    # deg imbalance
    pair_feat[:, 2] = fi[:, 1] + fj[:, 1]           # 2-hop sum
    pair_feat[:, 3] = (fi[:, 1] - fj[:, 1]) ** 2    # 2-hop imbalance
    pair_feat[:, 4] = fi[:, 2] + fj[:, 2]           # 3-hop sum
    pair_feat[:, 5] = (fi[:, 2] - fj[:, 2]) ** 2    # 3-hop imbalance
    return pair_feat


def compute_graph_features(
    n: int,
    m: int,
    step_frac: float,
    max_n: int,
) -> np.ndarray:
    """
    Compute 3-dim graph-level features. O(1).

    Features:
      [0] density   = 2m / (n(n-1))
      [1] step_frac = k / K_STEPS
      [2] n / max_n

    Returns:
        (3,) feature vector
    """
    features = np.zeros(GRAPH_FEAT_DIM, dtype=np.float32)
    features[0] = 2.0 * m / max(n * (n - 1), 1)
    features[1] = step_frac
    features[2] = n / max(max_n, 1)
    return features


def compute_phi(n: int, m: int) -> int:
    """φ = floor(√(m·(N_max−m)/N_max)), min 3. Tracks combinatorial space."""
    n_max = n * (n - 1) // 2
    return max(3, int(math.sqrt(m * (n_max - m) / n_max)))


def build_ring_initial(n: int, m: int) -> np.ndarray:
    """
    Build graph with m edges using deterministic ring-based construction.

    Algorithm (fully deterministic, parallelizable):
    - Add edges by offset: first all offset-1 edges, then offset-2, etc.
    - This creates a ring-like structure that expands outward uniformly.

    For m = n, this is exactly a ring. For m > n, it adds "shortcut" edges.

    Vectorized implementation for speed.
    """
    adj = np.zeros((n, n), dtype=np.float64)

    # Pre-compute all possible edges sorted by offset (ring distance)
    # This is deterministic and can be parallelized for large n
    edges_needed = m
    edges_added = 0

    for offset in range(1, (n + 1) // 2 + 1):
        if edges_added >= edges_needed:
            break

        # For this offset, edges are (i, (i+offset)%n) for all i
        # We only add where i < j to avoid duplicates

        # Vectorized edge generation for this offset
        i_arr = np.arange(n)
        j_arr = (i_arr + offset) % n

        # Filter to upper triangle (i < j)
        mask = i_arr < j_arr
        i_filtered = i_arr[mask]
        j_filtered = j_arr[mask]

        # How many can we add?
        can_add = min(len(i_filtered), edges_needed - edges_added)

        if can_add > 0:
            # Add edges (vectorized)
            adj[i_filtered[:can_add], j_filtered[:can_add]] = 1.0
            adj[j_filtered[:can_add], i_filtered[:can_add]] = 1.0
            edges_added += can_add

    return adj


def build_ring_initial_batch(n_list: List[int], m_list: List[int], num_workers: int = 4) -> List[np.ndarray]:
    """
    Build multiple graphs in parallel using thread pool.

    Args:
        n_list: List of node counts
        m_list: List of edge counts
        num_workers: Number of parallel workers

    Returns:
        List of adjacency matrices
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            lambda args: build_ring_initial(*args),
            zip(n_list, m_list)
        ))
    return results


def build_random_tree_initial(n: int, m: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Build random spanning tree + random residual edges.

    This creates a suboptimal graph that the agent can learn to improve.

    Algorithm:
    1. Random spanning tree via random permutation (n-1 edges)
    2. Add remaining (m - n + 1) edges randomly

    Args:
        n: Number of nodes
        m: Number of edges
        rng: Random state for reproducibility within episode

    Returns:
        Adjacency matrix (n, n)
    """
    adj = np.zeros((n, n), dtype=np.float64)

    # Step 1: Random spanning tree using Prüfer-like construction
    # Shuffle nodes and connect in chain with random branches
    nodes = list(range(n))
    rng.shuffle(nodes)

    # Connect nodes in shuffled order (creates a random tree)
    in_tree = {nodes[0]}
    tree_edges = 0

    for i in range(1, n):
        # Connect node to a random node already in tree
        new_node = nodes[i]
        existing = list(in_tree)
        parent = existing[rng.randint(len(existing))]

        adj[new_node, parent] = 1.0
        adj[parent, new_node] = 1.0
        in_tree.add(new_node)
        tree_edges += 1

    # Step 2: Add remaining edges randomly
    remaining = m - tree_edges

    if remaining > 0:
        # Get all possible non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))

        # Randomly select edges to add
        rng.shuffle(non_edges)
        for i, j in non_edges[:remaining]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj


def build_fv_initial(n: int, m: int) -> np.ndarray:
    """
    Build graph using Fiedler Vector greedy construction (matches C FV baseline).

    Algorithm:
    1. Ring graph (n edges, connectivity guaranteed)
    2. Greedily add non-edge with max |v₂ᵢ - v₂ⱼ|² until m edges

    O(M × N³) — negligible at training sizes n≤16.
    Deterministic given n, m (no randomness).
    """
    adj = np.zeros((n, n), dtype=np.float64)

    # Step 1: Ring (n edges)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    curr_m = n
    max_m = n * (n - 1) // 2
    target = min(m, max_m)

    while curr_m < target:
        # Compute Fiedler vector
        degrees = adj.sum(axis=1)
        L = np.diag(degrees) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]  # v₂

        # Find non-edge with max Fiedler gap
        best_score = -1.0
        best_u, best_v = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    diff_sq = (fiedler[i] - fiedler[j]) ** 2
                    if diff_sq > best_score:
                        best_score = diff_sq
                        best_u, best_v = i, j

        if best_u < 0:
            break

        adj[best_u, best_v] = 1.0
        adj[best_v, best_u] = 1.0
        curr_m += 1

    return adj


def build_ring_random_initial(n: int, m: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Build ring graph + random residual edges.

    Algorithm:
    1. Ring: connect node i to (i+1) % n for all i (n edges)
    2. Add remaining (m - n) edges randomly

    Matches the ring starting point of FV/ER baselines.

    Args:
        n: Number of nodes
        m: Number of edges
        rng: Random state for reproducibility within episode

    Returns:
        Adjacency matrix (n, n)
    """
    adj = np.zeros((n, n), dtype=np.float64)

    # Step 1: Ring (n edges)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    ring_edges = n

    # Step 2: Add remaining edges randomly
    remaining = m - ring_edges
    if remaining > 0:
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        rng.shuffle(non_edges)
        for i, j in non_edges[:remaining]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj


def build_random_k_regular(n: int, m: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Build random near-k-regular graph with m edges. Guarantees connectivity.

    Algorithm:
    1. Random Hamiltonian cycle (n edges, connectivity guaranteed, degree=2 for all)
    2. Add remaining edges with degree-balanced random selection
       (nodes below target degree k=floor(2m/n) are preferred)

    This provides a "hotter" start than random spanning trees by eliminating
    degree disparity, letting the RL agent focus on topology optimization.
    """
    adj = np.zeros((n, n), dtype=np.float64)
    k_target = (2.0 * m) / n

    # Phase 1: Random Hamiltonian cycle (guarantees connectivity)
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        u, v = perm[i], perm[(i + 1) % n]
        adj[u, v] = adj[v, u] = 1.0

    current_m = n

    if m < n:
        # m < n: remove edges from cycle while keeping connectivity
        cycle_edges = [(perm[i], perm[(i + 1) % n]) for i in range(n)]
        rng.shuffle(cycle_edges)
        for u, v in cycle_edges:
            if current_m <= m:
                break
            adj[u, v] = adj[v, u] = 0.0
            if is_connected(adj):
                current_m -= 1
            else:
                adj[u, v] = adj[v, u] = 1.0
        return adj

    # Phase 2: Add remaining edges, preferring low-degree nodes
    remaining = m - current_m
    if remaining > 0:
        degrees = adj.sum(axis=1)

        for _ in range(remaining):
            # Weighted random selection: higher deficit = more likely
            deficit = np.maximum(k_target - degrees, 0.1)
            probs = deficit / deficit.sum()
            u = rng.choice(n, p=probs)

            # Among non-neighbors of u, prefer those below target degree
            candidates = np.where((adj[u] == 0) & (np.arange(n) != u))[0]
            if len(candidates) == 0:
                # u connected to everyone — find another node with non-neighbors
                for alt_u in rng.permutation(n):
                    candidates = np.where((adj[alt_u] == 0) & (np.arange(n) != alt_u))[0]
                    if len(candidates) > 0:
                        u = int(alt_u)
                        break
                else:
                    break  # Graph is complete

            c_deficit = np.maximum(k_target - degrees[candidates], 0.1)
            c_probs = c_deficit / c_deficit.sum()
            v = candidates[rng.choice(len(candidates), p=c_probs)]

            adj[u, v] = adj[v, u] = 1.0
            degrees[u] += 1
            degrees[v] += 1

    return adj


@dataclass
class GNMConfig:
    """Configuration for G(n,m) environment."""
    min_n: int = 8
    max_n: int = 32

    # Full density range
    min_density: float = 0.0  # Will be clamped to n-1 edges minimum
    max_density: float = 1.0  # Up to complete graph


class GNMState(NamedTuple):
    """State for G(n,m) environment."""
    adj: np.ndarray
    n: int
    m: int
    current_round: int
    current_node: int
    total_rewires: int
    lambda2: float


class GNMEnv:
    """
    G(n,m) environment with edge-centric action space.

    Each step: pick ANY edge to remove + ANY non-edge to add (or KEEP).
    Total steps per episode = n × φ.
    """

    def __init__(
        self,
        config: GNMConfig,
        seed: Optional[int] = None,
        eval_mode: bool = False,
    ):
        self.config = config
        self.base_seed = seed
        self.eval_mode = eval_mode
        self.inference_mode = False
        self.episode_count = 0

        # State variables
        self.adj: Optional[np.ndarray] = None
        self.degrees: Optional[np.ndarray] = None
        self.n: int = 0
        self.m: int = 0
        self.total_rewires: int = 0
        self.lambda2: float = 0.0
        self.initial_lambda2: float = 0.0
        self._last_swap = None

    def reset(self, n: Optional[int] = None, m: Optional[int] = None) -> GNMState:
        """Reset environment with new (n, m) configuration."""
        self.episode_count += 1
        seed = self.base_seed + self.episode_count if self.base_seed else None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        if n is None:
            n = random.randint(self.config.min_n, self.config.max_n)
        if m is None:
            min_m = n - 1
            max_m = n * (n - 1) // 2
            m = random.randint(min_m, max_m)

        min_m = n - 1
        max_m = n * (n - 1) // 2
        m = max(min_m, min(m, max_m))

        self.n = n
        self.m = m

        # Random spanning tree + random edges
        self.adj = build_random_tree_initial(n, m, self.rng)
        self.degrees = self.adj.sum(axis=1)
        self._last_swap = None

        if self.inference_mode:
            # No eigensolver at inference — λ₂ unknown (not needed)
            self.lambda2 = 0.0
            self.initial_lambda2 = 0.0
        else:
            # Training: exact eigensolver for reward computation
            L = np.diag(self.degrees) - self.adj
            eigvals = np.linalg.eigvalsh(L)
            self.lambda2 = float(eigvals[1])
            self.initial_lambda2 = self.lambda2

        self.total_rewires = 0

        return self._get_state()

    def get_edge_index(self) -> np.ndarray:
        """Return (E, 2) array of edge indices (i < j)."""
        ei, ej = np.where(np.triu(self.adj, 1) > 0)
        return np.stack([ei, ej], axis=1)  # (E, 2)

    def get_non_edge_index(self) -> np.ndarray:
        """Return (NE, 2) array of non-edge indices (i < j)."""
        mask = np.triu(1.0 - self.adj - np.eye(self.n), 1) > 0
        ni, nj = np.where(mask)
        return np.stack([ni, nj], axis=1)  # (NE, 2)

    def step(self, remove_edge: Tuple[int, int], add_edge: Tuple[int, int]) -> Tuple[float, Dict]:
        """
        Execute edge swap: remove one edge, add one non-edge.

        The training loop manages episode structure (K steps).
        This method only performs the swap and returns per-step reward.

        Args:
            remove_edge: (u, v) edge to remove
            add_edge: (a, b) non-edge to add

        Returns:
            reward: per-step shaping reward (SHAPING_COEF * delta_lambda2)
            info: dict with swap details
        """
        u, v = remove_edge
        a, b = add_edge
        valid = self._try_swap(u, v, a, b)

        reward = 0.0
        if valid:
            self.total_rewires += 1
            if not self.inference_mode:
                # Training: exact eigensolver for per-step reward
                L = np.diag(self.degrees) - self.adj
                eigvals = np.linalg.eigvalsh(L)
                new_lambda2 = float(eigvals[1])
                reward = SHAPING_COEF * (new_lambda2 - self.lambda2)
                self.lambda2 = new_lambda2

        info = {
            'rewired': valid,
            'total_rewires': self.total_rewires,
            'n': self.n,
            'm': self.m,
        }

        return reward, info

    def batch_step(
        self,
        remove_edges: list,
        add_edges: list,
    ) -> Tuple[float, Dict]:
        """
        Apply batch of edge swaps: remove several edges, add several non-edges.

        Single BFS connectivity check. Revert ALL if disconnected.
        Single eigensolver call for reward.

        Args:
            remove_edges: list of (u, v) edges to remove
            add_edges: list of (a, b) non-edges to add

        Returns:
            reward: shaping reward (SHAPING_COEF * delta_lambda2)
            info: dict with swap details
        """
        R = len(remove_edges)

        if R == 0:
            return 0.0, {
                'rewired': 0, 'total_rewires': self.total_rewires,
                'n': self.n, 'm': self.m, 'reverted': False,
            }

        # Save state for potential revert
        saved_adj = self.adj.copy()
        saved_degrees = self.degrees.copy()

        # Apply all removes
        for u, v in remove_edges:
            self.adj[u, v] = self.adj[v, u] = 0.0

        # Apply adds (cap at R to maintain m)
        actual_adds = min(len(add_edges), R)
        for i in range(actual_adds):
            a, b = add_edges[i]
            self.adj[a, b] = self.adj[b, a] = 1.0

        self.degrees = self.adj.sum(axis=1)

        # Single BFS connectivity check
        if not is_connected(self.adj):
            self.adj = saved_adj
            self.degrees = saved_degrees
            return 0.0, {
                'rewired': 0, 'total_rewires': self.total_rewires,
                'n': self.n, 'm': self.m, 'reverted': True,
            }

        self.total_rewires += actual_adds

        reward = 0.0
        if not self.inference_mode:
            # Training: exact eigensolver for reward
            L = np.diag(self.degrees) - self.adj
            eigvals = np.linalg.eigvalsh(L)
            new_lambda2 = float(eigvals[1])
            delta = new_lambda2 - self.lambda2
            reward = max(0.0, delta)  # Positive-only: no penalty for bad rewires
            self.lambda2 = new_lambda2

        return reward, {
            'rewired': actual_adds, 'total_rewires': self.total_rewires,
            'n': self.n, 'm': self.m, 'reverted': False,
        }

    def _try_swap(self, u: int, v: int, a: int, b: int) -> bool:
        """Try to swap: remove edge (u,v), add edge (a,b). Returns True if successful."""
        if self.adj[u, v] == 0:
            return False
        if self.adj[a, b] > 0:
            return False

        self.adj[u, v] = self.adj[v, u] = 0.0
        self.adj[a, b] = self.adj[b, a] = 1.0

        if not is_connected(self.adj):
            self.adj[u, v] = self.adj[v, u] = 1.0
            self.adj[a, b] = self.adj[b, a] = 0.0
            return False

        self.degrees[u] -= 1
        self.degrees[v] -= 1
        self.degrees[a] += 1
        self.degrees[b] += 1

        self._last_swap = (u, v, a, b)
        return True

    def spectral_step(
        self,
        remove_edges: List[Tuple[int, int]],
        add_edges: List[Tuple[int, int]],
    ) -> Tuple[float, Dict]:
        """
        Apply all edge swaps in batch. No BFS, no bridge detection.

        Training: eigensolver detects disconnection (λ₂ < ε) → revert + heavy penalty.
                  The agent learns to avoid disconnecting moves.
        Inference: no checks, trust the learned policy.

        Args:
            remove_edges: list of (u, v) edges to remove
            add_edges: list of (a, b) non-edges to add

        Returns:
            reward: Δλ₂ or DISCONNECT_PENALTY
            info: dict with details
        """
        R = min(len(remove_edges), len(add_edges))

        if R == 0:
            return 0.0, {
                'successful_swaps': 0,
                'attempted_swaps': 0,
                'total_rewires': self.total_rewires,
                'n': self.n,
                'm': self.m,
                'disconnected': False,
            }

        # Save state for potential revert (training only)
        if not self.inference_mode:
            saved_adj = self.adj.copy()
            saved_degrees = self.degrees.copy()

        # Apply all swaps in batch — O(R)
        applied = 0
        for i in range(R):
            u, v = remove_edges[i]
            a, b = add_edges[i]

            if self.adj[u, v] == 0 or self.adj[a, b] > 0:
                continue

            self.adj[u, v] = self.adj[v, u] = 0.0
            self.adj[a, b] = self.adj[b, a] = 1.0
            self.degrees[u] -= 1
            self.degrees[v] -= 1
            self.degrees[a] += 1
            self.degrees[b] += 1
            applied += 1

        self.total_rewires += applied

        if applied == 0:
            return 0.0, {
                'successful_swaps': 0,
                'attempted_swaps': R,
                'total_rewires': self.total_rewires,
                'n': self.n,
                'm': self.m,
                'disconnected': False,
            }

        reward = 0.0
        disconnected = False

        if not self.inference_mode:
            # Training: eigensolver for reward (O(N³) allowed)
            L = np.diag(self.degrees) - self.adj
            eigvals = np.linalg.eigvalsh(L)
            new_lambda2 = float(eigvals[1])

            if new_lambda2 < 1e-6:
                # Disconnected! Revert and penalize heavily.
                self.adj = saved_adj
                self.degrees = saved_degrees
                self.total_rewires -= applied
                reward = DISCONNECT_PENALTY
                disconnected = True
            else:
                reward = new_lambda2 - self.lambda2
                self.lambda2 = new_lambda2
        # Inference: no eigensolver, no check — trust the policy

        return reward, {
            'successful_swaps': 0 if disconnected else applied,
            'attempted_swaps': R,
            'total_rewires': self.total_rewires,
            'n': self.n,
            'm': self.m,
            'disconnected': disconnected,
        }

    def _get_state(self) -> GNMState:
        return GNMState(
            adj=self.adj.copy() if self.adj is not None else np.array([]),
            n=self.n,
            m=self.m,
            current_round=0,
            current_node=0,
            total_rewires=self.total_rewires,
            lambda2=self.lambda2,
        )


def compute_R(n: int, m: int) -> int:
    """Compute R = min(m//4, NE) — number of swaps per step, scales with edge count."""
    ne = n * (n - 1) // 2 - m
    return max(1, min(m // 4, ne))


class BatchedGNMEnv:
    """
    Batched G(n,m) environment for GPU-parallel training.

    Each graph in the batch has its own (n, m). Tensors are padded to max_n.
    node_mask (B, max_n) tracks which nodes are real vs padding.
    """

    def __init__(self, batch_size: int, min_n: int, max_n: int, device: str):
        self.batch_size = batch_size
        self.min_n = min_n
        self.max_n = max_n
        self.device = torch.device(device)
        self.use_cuda_kernels = (_gnm_cuda is not None and self.device.type == 'cuda')
        self.inference_mode = False
        self.max_steps = 0
        self._cached_nb_mask = None
        self._cached_dest_mask = None

    def _eigsolve(self, L: torch.Tensor):
        """Batched eigsolve with CPU fallback for MPS/unsupported devices."""
        try:
            return torch.linalg.eigh(L)
        except (NotImplementedError, RuntimeError):
            eigvals, eigvecs = torch.linalg.eigh(L.cpu())
            return eigvals.to(self.device), eigvecs.to(self.device)

    def reset(self, n: Optional[int] = None, m: Optional[int] = None):
        """Reset all B envs. Each gets its own (n, m) unless n/m are fixed."""
        B = self.batch_size
        N = self.max_n

        # Per-graph n: either fixed or random
        if n is not None:
            self.n_per_graph = torch.full((B,), n, dtype=torch.long)
        else:
            self.n_per_graph = torch.randint(self.min_n, self.max_n + 1, (B,))

        # Per-graph m: fixed, or random in [n-1, n*(n-1)/2]
        self.m = torch.zeros(B, dtype=torch.long)
        if m is not None:
            self.m[:] = m
        else:
            for b in range(B):
                n_b = self.n_per_graph[b].item()
                min_m = n_b - 1
                max_m = n_b * (n_b - 1) // 2
                self.m[b] = random.randint(min_m, max_m)

        # phi = floor(sqrt(m*(N_max-m)/N_max)), clamped to >= 3
        n_max = self.n_per_graph * (self.n_per_graph - 1) // 2
        self.phi = torch.clamp(
            torch.sqrt((self.m * (n_max - self.m)).float() / n_max.float()).long(),
            min=3,
        ).to(self.device)

        # Node mask: True for real nodes (i < n_b)
        arange = torch.arange(N)
        self.node_mask = (arange.unsqueeze(0) < self.n_per_graph.unsqueeze(1)).to(self.device)

        if self.use_cuda_kernels:
            m_vals_gpu = self.m.to(dtype=torch.int32, device=self.device)
            n_vals_gpu = self.n_per_graph.to(dtype=torch.int32, device=self.device)
            self.n_vals_i32 = n_vals_gpu
            seed = random.randint(0, 2**62)
            self.adj, self.degrees = _gnm_cuda.build_random_graphs(
                m_vals_gpu, n_vals_gpu, B, N, seed
            )

            L = torch.diag_embed(self.degrees) - self.adj
            L.diagonal(dim1=-2, dim2=-1)[~self.node_mask] = 1e6
            evals, evecs = _gnm_cuda.batched_jacobi_eigh(L, 10)
            sorted_evals, sorted_idx = evals.sort(dim=-1)
            self.lambda2 = sorted_evals[:, 1]
            fiedler_col = sorted_idx[:, 1]
            col_idx = fiedler_col.view(B, 1, 1).expand(B, N, 1)
            self.fiedler = evecs.gather(2, col_idx).squeeze(2)
            self.fiedler[~self.node_mask] = 0.0

            self.phi_i32 = self.phi.to(torch.int32)
            self.current_node = torch.zeros(B, dtype=torch.int32, device=self.device)
            self.current_round = torch.zeros(B, dtype=torch.int32, device=self.device)
            self.done_i32 = torch.zeros(B, dtype=torch.int32, device=self.device)
            self.total_rewires = torch.zeros(B, dtype=torch.int32, device=self.device)
            self.done = torch.zeros(B, dtype=torch.bool, device=self.device)
        else:
            adjs = np.zeros((B, N, N), dtype=np.float64)
            for b in range(B):
                n_b = self.n_per_graph[b].item()
                m_b = self.m[b].item()
                rng = np.random.RandomState()
                adj_b = build_random_tree_initial(n_b, m_b, rng)
                adjs[b, :n_b, :n_b] = adj_b
            self.adj = torch.tensor(adjs, dtype=torch.float32, device=self.device)
            self.degrees = self.adj.sum(dim=-1)

            L = torch.diag_embed(self.degrees) - self.adj
            L.diagonal(dim1=-2, dim2=-1)[~self.node_mask] = 1e6
            eigvals, eigvecs = self._eigsolve(L)
            self.lambda2 = eigvals[:, 1]
            self.fiedler = eigvecs[:, :, 1]
            self.fiedler[~self.node_mask] = 0.0

            self.current_node = torch.zeros(B, dtype=torch.long, device=self.device)
            self.current_round = torch.zeros(B, dtype=torch.long, device=self.device)
            self.done = torch.zeros(B, dtype=torch.bool, device=self.device)
            self.total_rewires = torch.zeros(B, dtype=torch.long, device=self.device)

        sign = torch.sign(self.fiedler.sum(dim=-1, keepdim=True))
        sign[sign == 0] = 1.0
        self.fiedler = self.fiedler * sign

        self.initial_lambda2 = self.lambda2.clone()

        # max_steps = max(n_b * phi_b) across batch (one-time sync)
        steps_per_graph = self.n_per_graph.to(self.device) * self.phi
        self.max_steps = int(steps_per_graph.max().item())

    def compute_features(self) -> torch.Tensor:
        """Compute (B, max_n, 5) node features. Padding nodes get zeros."""
        B, N, _ = self.adj.shape
        batch_idx = torch.arange(B, device=self.device)

        if self.use_cuda_kernels:
            feat, nb_mask, dest_mask = _gnm_cuda.fused_features_and_masks(
                self.adj, self.degrees, self.fiedler,
                self.current_node,
                self.n_vals_i32,
                N
            )
            self._cached_nb_mask = nb_mask > 0.5
            self._cached_dest_mask = dest_mask > 0.5
            # CUDA kernel returns 4-dim; append round_frac as 5th feature
            round_frac = (self.current_round.float() / self.phi.float().clamp(min=1))
            round_feat = round_frac.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1)
            feat = torch.cat([feat, round_feat], dim=-1)
            feat[~self.node_mask] = 0.0
            return feat

        feat = torch.zeros(B, N, NODE_FEAT_DIM, device=self.device)
        n_float = self.n_per_graph.to(device=self.device, dtype=torch.float32)

        feat[:, :, 0] = self.degrees / (n_float.unsqueeze(-1) - 1).clamp(min=1)
        feat[:, :, 1] = self.fiedler
        u_fiedler = self.fiedler[batch_idx, self.current_node]
        feat[:, :, 2] = (u_fiedler.unsqueeze(-1) - self.fiedler) ** 2
        feat[batch_idx, self.current_node, 3] = 1.0
        round_frac = (self.current_round.float() / self.phi.float().clamp(min=1))
        feat[:, :, 4] = round_frac.unsqueeze(-1)

        feat[~self.node_mask] = 0.0
        return feat

    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get neighbor_mask and dest_mask, each (B, max_n). Padding = False."""
        if self.use_cuda_kernels and self._cached_nb_mask is not None:
            return self._cached_nb_mask, self._cached_dest_mask

        B = self.adj.shape[0]
        batch_idx = torch.arange(B, device=self.device)
        u_rows = self.adj[batch_idx, self.current_node]
        neighbor_mask = u_rows > 0
        dest_mask = (u_rows == 0) & self.node_mask
        dest_mask[batch_idx, self.current_node] = False
        return neighbor_mask, dest_mask

    def step(self, neighbor_actions: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:
        """
        Execute batched rewiring step.

        Args:
            neighbor_actions: (B,) neighbor to drop, >= max_n means KEEP
            destinations: (B,) new destination node

        Returns:
            rewards: (B,) per-step delta-lambda2
        """
        B, N, _ = self.adj.shape

        if self.use_cuda_kernels:
            na_i32 = neighbor_actions.to(torch.int32)
            dst_i32 = destinations.to(torch.int32)
            was_done = self.done_i32.bool().clone()
            rewards = _gnm_cuda.fused_step(
                self.adj, self.degrees, self.lambda2, self.fiedler,
                na_i32, dst_i32,
                self.current_node, self.current_round,
                self.phi_i32, self.done_i32, self.total_rewires,
                self.n_vals_i32,
                N, 10, 1 if self.inference_mode else 0
            )
            self.done = self.done_i32.bool()
            if not self.inference_mode:
                rewards = rewards * SHAPING_COEF
                newly_done = self.done & ~was_done
                if newly_done.any():
                    rewards[newly_done] += (self.lambda2 - self.initial_lambda2)[newly_done]
            self._cached_nb_mask = None
            self._cached_dest_mask = None
            return rewards

        rewards = torch.zeros(B, device=self.device)
        n_per = self.n_per_graph.to(self.device)

        active = ~self.done
        # KEEP if action >= max_n (policy KEEP index)
        is_keep = neighbor_actions >= N
        rewiring = active & ~is_keep

        if rewiring.any():
            rew_idx = rewiring.nonzero(as_tuple=True)[0]
            u = self.current_node[rew_idx]
            v = neighbor_actions[rew_idx]
            w = destinations[rew_idx]

            valid = (v != w) & (u != w)
            valid &= (self.adj[rew_idx, u, v] == 1)
            valid &= (self.adj[rew_idx, u, w] == 0)

            if valid.any():
                vs = valid.nonzero(as_tuple=True)[0]
                vi = rew_idx[vs]
                vu, vv, vw = u[vs], v[vs], w[vs]

                old_lambda2 = self.lambda2.clone()
                old_fiedler = self.fiedler.clone()

                self.adj[vi, vu, vv] = 0
                self.adj[vi, vv, vu] = 0
                self.adj[vi, vu, vw] = 1
                self.adj[vi, vw, vu] = 1
                self.degrees[vi] = self.adj[vi].sum(dim=-1)

                L = torch.diag_embed(self.degrees) - self.adj
                L.diagonal(dim1=-2, dim2=-1)[~self.node_mask] = 1e6
                eigvals, eigvecs = self._eigsolve(L)
                new_lambda2 = eigvals[:, 1]
                new_fiedler = eigvecs[:, :, 1]

                dot = (new_fiedler * old_fiedler).sum(dim=-1, keepdim=True)
                sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
                new_fiedler = new_fiedler * sign

                disconnected = new_lambda2[vi] < 1e-6

                if disconnected.any():
                    di = vi[disconnected]
                    du, dv, dw = vu[disconnected], vv[disconnected], vw[disconnected]
                    self.adj[di, du, dv] = 1
                    self.adj[di, dv, du] = 1
                    self.adj[di, du, dw] = 0
                    self.adj[di, dw, du] = 0
                    self.degrees[di] = self.adj[di].sum(dim=-1)
                    new_lambda2[di] = old_lambda2[di]
                    new_fiedler[di] = old_fiedler[di]

                success = torch.zeros(B, dtype=torch.bool, device=self.device)
                success[vi] = ~disconnected

                self.lambda2 = torch.where(success, new_lambda2, old_lambda2)
                self.fiedler = torch.where(
                    success.unsqueeze(-1), new_fiedler, old_fiedler
                )
                if self.inference_mode:
                    rewards[success] = new_lambda2[success] - old_lambda2[success]
                else:
                    rewards[success] = SHAPING_COEF * (new_lambda2[success] - old_lambda2[success])
                self.total_rewires[success] += 1

        # Advance current_node for active envs
        self.current_node[active] += 1

        # Round advancement: node >= n_per_graph[b] means round complete
        round_done = (self.current_node >= n_per) & active
        self.current_round[round_done] += 1
        self.current_node[round_done] = 0

        was_done = self.done.clone()
        self.done |= (self.current_round >= self.phi)
        if not self.inference_mode:
            newly_done = self.done & ~was_done
            if newly_done.any():
                rewards[newly_done] += (self.lambda2 - self.initial_lambda2)[newly_done]

        return rewards

    def all_done(self) -> bool:
        """Check if all envs in batch have finished."""
        return self.done.all().item()
