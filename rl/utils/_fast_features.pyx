# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated graph feature computation.

Drop-in replacements for pure Python functions in spectral.py.
All functions maintain O(N²) complexity for inference.
"""

import numpy as np
cimport numpy as np
from libc.math cimport log
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t FTYPE_t
ctypedef np.int32_t ITYPE_t


def bfs_distances(double[:, :] adj not None, int source):
    """
    BFS distances from source to all nodes. O(N+M).

    Args:
        adj: (N, N) adjacency matrix (dense, float64)
        source: Source node index

    Returns:
        (N,) int32 distance array. Unreachable nodes get distance n.
    """
    cdef int n = adj.shape[0]
    cdef np.ndarray[ITYPE_t, ndim=1] dist = np.full(n, n, dtype=np.int32)
    cdef int* queue = <int*>malloc(n * sizeof(int))
    cdef int head = 0, tail = 0
    cdef int node, nb, d

    if queue == NULL:
        raise MemoryError()

    try:
        dist[source] = 0
        queue[0] = source
        tail = 1

        while head < tail:
            node = queue[head]
            head += 1
            d = dist[node] + 1
            for nb in range(n):
                if adj[node, nb] > 0 and dist[nb] == n:
                    dist[nb] = d
                    queue[tail] = nb
                    tail += 1
    finally:
        free(queue)

    return np.asarray(dist)


def is_connected_fast(double[:, :] adj not None):
    """
    Check if graph is connected using BFS. O(N+M).

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        True if connected, False otherwise
    """
    cdef int n = adj.shape[0]
    if n == 0:
        return True

    cdef int* queue = <int*>malloc(n * sizeof(int))
    cdef char* visited = <char*>malloc(n * sizeof(char))
    cdef int head = 0, tail = 0, count = 1
    cdef int node, nb
    cdef bint result

    if queue == NULL or visited == NULL:
        if queue != NULL:
            free(queue)
        if visited != NULL:
            free(visited)
        raise MemoryError()

    try:
        memset(visited, 0, n * sizeof(char))
        visited[0] = 1
        queue[0] = 0
        tail = 1

        while head < tail:
            node = queue[head]
            head += 1
            for nb in range(n):
                if adj[node, nb] > 0 and visited[nb] == 0:
                    visited[nb] = 1
                    queue[tail] = nb
                    tail += 1
                    count += 1

        result = (count == n)
    finally:
        free(queue)
        free(visited)

    return result


def compute_rwlp_features(double[:, :] adj not None):
    """
    O(N²) Random Walk spectral proxy features.

    Returns (N, 4) float32 array:
        0: log(diag(P²) * n) — 2-step return probability
        1: log(mean neighbor P² return * n) — neighbor spectral quality
        2: mean neighbor degree / max_degree — local density
        3: inverse-degree neighbor sum (normalized) — resistance proxy
    """
    cdef int n = adj.shape[0]
    cdef int i, j

    # Compute degrees
    cdef double* degrees = <double*>malloc(n * sizeof(double))
    cdef double* d_inv = <double*>malloc(n * sizeof(double))
    if degrees == NULL or d_inv == NULL:
        if degrees != NULL: free(degrees)
        if d_inv != NULL: free(d_inv)
        raise MemoryError()

    cdef double max_deg = 1.0
    for i in range(n):
        degrees[i] = 0.0
        for j in range(n):
            degrees[i] += adj[i, j]
        if degrees[i] > 0:
            d_inv[i] = 1.0 / degrees[i]
        else:
            d_inv[i] = 0.0
        if degrees[i] > max_deg:
            max_deg = degrees[i]

    # Build P and compute diag(P²) = sum_j P[i,j]*P[j,i] — O(N²)
    # P[i,j] = adj[i,j] / deg[i], P[j,i] = adj[j,i] / deg[j]
    # Since adj symmetric: P[i,j]*P[j,i] = adj[i,j]² * d_inv[i] * d_inv[j]
    #                                     = adj[i,j] * d_inv[i] * d_inv[j] (binary adj)
    cdef double* p2_diag = <double*>malloc(n * sizeof(double))
    if p2_diag == NULL:
        free(degrees); free(d_inv)
        raise MemoryError()

    for i in range(n):
        p2_diag[i] = 0.0
        for j in range(n):
            if adj[i, j] > 0:
                p2_diag[i] += d_inv[i] * d_inv[j]

    cdef np.ndarray[FTYPE_t, ndim=2] features = np.zeros((n, 4), dtype=np.float32)
    cdef double val, max_ids

    # Feature 0: log(diag(P²) * n)
    for i in range(n):
        features[i, 0] = <float>log(p2_diag[i] * n + 1e-10)

    # Feature 1: mean neighbor P² return — O(N²)
    for i in range(n):
        if degrees[i] > 0:
            val = 0.0
            for j in range(n):
                if adj[i, j] > 0:
                    val += p2_diag[j]
            val *= d_inv[i]
            features[i, 1] = <float>log(val * n + 1e-10)
        else:
            features[i, 1] = <float>log(1e-10)

    # Feature 2: mean neighbor degree / max_degree — O(N²)
    for i in range(n):
        if degrees[i] > 0:
            val = 0.0
            for j in range(n):
                if adj[i, j] > 0:
                    val += degrees[j]
            features[i, 2] = <float>(val * d_inv[i] / max_deg)

    # Feature 3: inverse-degree neighbor sum (normalized) — O(N²)
    cdef double* inv_deg_sums = <double*>malloc(n * sizeof(double))
    if inv_deg_sums == NULL:
        free(degrees); free(d_inv); free(p2_diag)
        raise MemoryError()

    max_ids = 1e-10
    for i in range(n):
        inv_deg_sums[i] = 0.0
        for j in range(n):
            if adj[i, j] > 0:
                inv_deg_sums[i] += d_inv[j]
        if inv_deg_sums[i] > max_ids:
            max_ids = inv_deg_sums[i]

    for i in range(n):
        features[i, 3] = <float>(inv_deg_sums[i] / max_ids)

    free(degrees)
    free(d_inv)
    free(p2_diag)
    free(inv_deg_sums)

    return np.asarray(features)


def find_bridges_fast(double[:, :] adj not None):
    """
    Find all bridge edges using iterative Tarjan's algorithm. O(N+M).

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        Set of (min(u,v), max(u,v)) tuples for each bridge edge.
    """
    cdef int n = adj.shape[0]
    if n <= 1:
        return set()

    cdef int* disc = <int*>malloc(n * sizeof(int))
    cdef int* low = <int*>malloc(n * sizeof(int))
    cdef int* parent = <int*>malloc(n * sizeof(int))
    # Stack: each entry is (node, neighbor_index) stored as 2 ints
    cdef int* stack_node = <int*>malloc(n * sizeof(int))
    cdef int* stack_ni = <int*>malloc(n * sizeof(int))
    # Neighbor lists: for each node, store sorted neighbor indices
    cdef int* nb_counts = <int*>malloc(n * sizeof(int))
    # Flat neighbor array: max n*n entries (but typically much less)
    cdef int* nb_flat = <int*>malloc(n * n * sizeof(int))

    if (disc == NULL or low == NULL or parent == NULL or
        stack_node == NULL or stack_ni == NULL or
        nb_counts == NULL or nb_flat == NULL):
        if disc != NULL: free(disc)
        if low != NULL: free(low)
        if parent != NULL: free(parent)
        if stack_node != NULL: free(stack_node)
        if stack_ni != NULL: free(stack_ni)
        if nb_counts != NULL: free(nb_counts)
        if nb_flat != NULL: free(nb_flat)
        raise MemoryError()

    cdef int i, j, u, v, ni, timer, sp, p
    cdef int nb_offset
    cdef bint found_child

    bridges = set()

    try:
        # Initialize
        for i in range(n):
            disc[i] = -1
            low[i] = -1
            parent[i] = -1

        # Pre-build adjacency lists
        nb_offset = 0
        for i in range(n):
            nb_counts[i] = 0
            for j in range(n):
                if adj[i, j] > 0:
                    nb_flat[i * n + nb_counts[i]] = j
                    nb_counts[i] += 1

        timer = 0

        for start in range(n):
            if disc[start] != -1:
                continue

            # Push start node
            sp = 0
            stack_node[0] = start
            stack_ni[0] = 0
            disc[start] = timer
            low[start] = timer
            timer += 1

            while sp >= 0:
                u = stack_node[sp]
                ni = stack_ni[sp]
                found_child = False

                while ni < nb_counts[u]:
                    v = nb_flat[u * n + ni]
                    ni += 1
                    stack_ni[sp] = ni  # Update neighbor index

                    if disc[v] == -1:
                        # Tree edge: push child
                        parent[v] = u
                        disc[v] = timer
                        low[v] = timer
                        timer += 1
                        sp += 1
                        stack_node[sp] = v
                        stack_ni[sp] = 0
                        found_child = True
                        break
                    elif v != parent[u]:
                        # Back edge: update low
                        if disc[v] < low[u]:
                            low[u] = disc[v]

                if not found_child:
                    # Done with u — pop and update parent's low
                    sp -= 1
                    if sp >= 0:
                        p = stack_node[sp]
                        if low[u] < low[p]:
                            low[p] = low[u]
                        # Bridge condition
                        if low[u] > disc[p]:
                            if p < u:
                                bridges.add((p, u))
                            else:
                                bridges.add((u, p))
    finally:
        free(disc)
        free(low)
        free(parent)
        free(stack_node)
        free(stack_ni)
        free(nb_counts)
        free(nb_flat)

    return bridges


def compute_anchor_distances(double[:, :] adj not None):
    """
    BFS distances to 3 anchor nodes (normalized). O(N+M).

    Anchors:
        0: node 0
        1: farthest from node 0
        2: farthest from anchor 1

    Returns (N, 3) float32 array.
    """
    cdef int n = adj.shape[0]
    cdef int i

    cdef np.ndarray[ITYPE_t, ndim=1] d0 = bfs_distances(adj, 0)

    # Find farthest from 0
    cdef int anchor1 = 0
    cdef int max_d = 0
    for i in range(n):
        if d0[i] < n and d0[i] > max_d:
            max_d = d0[i]
            anchor1 = i

    cdef np.ndarray[ITYPE_t, ndim=1] d1 = bfs_distances(adj, anchor1)

    # Find farthest from anchor1
    cdef int anchor2 = 0
    max_d = 0
    for i in range(n):
        if d1[i] < n and d1[i] > max_d:
            max_d = d1[i]
            anchor2 = i

    cdef np.ndarray[ITYPE_t, ndim=1] d2 = bfs_distances(adj, anchor2)

    # Normalize and build output
    cdef np.ndarray[FTYPE_t, ndim=2] features = np.zeros((n, 3), dtype=np.float32)
    cdef float norm0 = max(<float>np.max(d0), 1.0)
    cdef float norm1 = max(<float>np.max(d1), 1.0)
    cdef float norm2 = max(<float>np.max(d2), 1.0)

    for i in range(n):
        features[i, 0] = <float>d0[i] / norm0
        features[i, 1] = <float>d1[i] / norm1
        features[i, 2] = <float>d2[i] / norm2

    return np.asarray(features)
