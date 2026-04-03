"""
OURS - O(N²) Deterministic Graph Builder

Builds high-connectivity graphs in O(N²) time without eigensolvers.
Used as Phase 1 of the Deterministic + RL pipeline.

Algorithm:
1. Build random spanning tree (ensures connectivity)
2. Greedily add edges from min-degree nodes to low-degree non-neighbors
3. Use bucket queue for O(1) min-degree lookup
4. Use per-node non-edge lists for O(1) non-neighbor sampling

Complexity:
- Space: O(N²) for non-edge tracking
- Time: O(N²) initialization + O(M) edge additions = O(N²) total
"""

import numpy as np
from typing import Tuple, List, Optional
import random


class BucketQueue:
    """O(1) amortized min-degree lookup using bucket queue."""

    def __init__(self, n: int):
        self.n = n
        self.degrees = np.zeros(n, dtype=np.int32)
        self.buckets: List[set] = [set() for _ in range(n)]
        self.min_degree = 0

        # All nodes start at degree 0
        self.buckets[0] = set(range(n))

    def get_min_degree_node(self) -> int:
        """Get any node with minimum degree. O(1) amortized."""
        while self.min_degree < self.n and len(self.buckets[self.min_degree]) == 0:
            self.min_degree += 1

        if self.min_degree >= self.n:
            return -1

        # Return arbitrary node from min bucket
        return next(iter(self.buckets[self.min_degree]))

    def increase_degree(self, node: int):
        """Move node to next degree bucket. O(1)."""
        old_deg = self.degrees[node]
        self.buckets[old_deg].discard(node)

        new_deg = old_deg + 1
        if new_deg < self.n:
            self.buckets[new_deg].add(node)
        self.degrees[node] = new_deg

    def get_degree(self, node: int) -> int:
        return self.degrees[node]


class NonEdgeTracker:
    """
    Track non-edges per node for O(1) random non-neighbor sampling.

    Space: O(N²)
    Operations: O(1) sample, O(1) remove
    """

    def __init__(self, n: int):
        self.n = n
        # For each node, list of non-neighbors
        self.nonedges: List[List[int]] = [list(range(n)) for _ in range(n)]
        # Remove self from each list
        for u in range(n):
            self.nonedges[u].remove(u)

        # Position index: position[u][v] = index of v in nonedges[u], or -1
        self.position = np.full((n, n), -1, dtype=np.int32)
        for u in range(n):
            for idx, v in enumerate(self.nonedges[u]):
                self.position[u, v] = idx

    def remove(self, u: int, v: int):
        """Remove edge (u,v) from non-edge sets. O(1) via swap-with-last."""
        self._remove_one(u, v)
        self._remove_one(v, u)

    def _remove_one(self, u: int, v: int):
        """Remove v from u's non-neighbor list."""
        idx = self.position[u, v]
        if idx < 0:
            return  # Already removed

        # Swap with last element
        last_idx = len(self.nonedges[u]) - 1
        if idx != last_idx:
            last_v = self.nonedges[u][last_idx]
            self.nonedges[u][idx] = last_v
            self.position[u, last_v] = idx

        self.nonedges[u].pop()
        self.position[u, v] = -1

    def sample_nonedge(self, u: int) -> int:
        """Sample random non-neighbor of u. O(1)."""
        if len(self.nonedges[u]) == 0:
            return -1
        return random.choice(self.nonedges[u])

    def count(self, u: int) -> int:
        return len(self.nonedges[u])

    def get_all(self, u: int) -> List[int]:
        return self.nonedges[u]


def ours_build(n: int, m: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Build a graph with n nodes and m edges using OURS algorithm.

    Optimizes for high algebraic connectivity without computing eigenvalues.
    Runs in O(N²) time.

    Args:
        n: Number of nodes
        m: Target number of edges
        seed: Random seed (optional)

    Returns:
        adj: (n, n) adjacency matrix
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adj = np.zeros((n, n), dtype=np.float64)
    bucket_queue = BucketQueue(n)
    nonedge_tracker = NonEdgeTracker(n)
    edge_count = 0

    def add_edge(u: int, v: int) -> bool:
        nonlocal edge_count
        if adj[u, v] > 0:
            return False

        adj[u, v] = adj[v, u] = 1.0
        edge_count += 1

        # Update data structures
        bucket_queue.increase_degree(u)
        bucket_queue.increase_degree(v)
        nonedge_tracker.remove(u, v)

        return True

    # Phase 1: Build random spanning tree (n-1 edges)
    # Each node connects to a random earlier node
    for i in range(1, n):
        if edge_count >= m:
            break
        target = random.randint(0, i - 1)
        add_edge(i, target)

    # Phase 2: Add edges from min-degree nodes
    # This promotes degree regularity (good for algebraic connectivity)
    max_attempts = m * 3  # Safety limit
    attempts = 0

    while edge_count < m and attempts < max_attempts:
        attempts += 1

        # Get min-degree node
        u = bucket_queue.get_min_degree_node()
        if u < 0:
            break

        # Sample non-neighbors and pick best one
        # Best = lowest degree with no/few common neighbors
        best_v = -1
        best_score = float('-inf')

        num_samples = min(32, nonedge_tracker.count(u))
        if num_samples == 0:
            continue

        candidates = random.sample(nonedge_tracker.get_all(u), num_samples)

        for v in candidates:
            v_deg = bucket_queue.get_degree(v)
            u_deg = bucket_queue.get_degree(u)

            # Score: prefer low-degree targets, penalize imbalance
            score = -v_deg * 3 - abs(u_deg - v_deg)

            if score > best_score:
                best_score = score
                best_v = v

            # Early exit: found good target
            if v_deg <= u_deg + 1:
                best_v = v
                break

        if best_v >= 0:
            add_edge(u, best_v)
        else:
            # Fallback: random non-neighbor
            v = nonedge_tracker.sample_nonedge(u)
            if v >= 0:
                add_edge(u, v)

    return adj


def ours_build_batch(configs: List[Tuple[int, int]], seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Build multiple graphs in batch.

    Args:
        configs: List of (n, m) tuples
        seed: Base random seed

    Returns:
        List of adjacency matrices
    """
    results = []
    for i, (n, m) in enumerate(configs):
        s = seed + i if seed is not None else None
        results.append(ours_build(n, m, seed=s))
    return results
