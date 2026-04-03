"""
Graph6 encoding/decoding utilities.

Graph6 is a compact ASCII encoding for undirected graphs.
- Compact: O(N²/6) bytes vs O(N²) for adjacency matrix
- Portable: ASCII string, easy to store/transfer
- Standard: Widely supported (networkx, nauty, etc.)
"""

import numpy as np
import torch
from typing import Union


def adj_to_graph6(adj: Union[np.ndarray, torch.Tensor]) -> str:
    """
    Convert adjacency matrix to Graph6 format.

    Args:
        adj: (N, N) adjacency matrix (numpy or torch)

    Returns:
        g6: Graph6 encoded string
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()

    adj = (adj > 0).astype(np.uint8)
    n = adj.shape[0]

    # Encode n
    if n <= 62:
        n_bytes = bytes([n + 63])
    elif n <= 258047:
        n_bytes = bytes([126, (n >> 12) + 63, ((n >> 6) & 63) + 63, (n & 63) + 63])
    else:
        raise ValueError(f"Graph too large for Graph6: n={n}")

    # Encode adjacency (upper triangle, row by row)
    bits = []
    for j in range(1, n):
        for i in range(j):
            bits.append(adj[i, j])

    # Pad to multiple of 6
    while len(bits) % 6 != 0:
        bits.append(0)

    # Convert bits to bytes
    adj_bytes = bytearray()
    for i in range(0, len(bits), 6):
        byte_val = 0
        for j in range(6):
            byte_val = (byte_val << 1) | bits[i + j]
        adj_bytes.append(byte_val + 63)

    return (n_bytes + bytes(adj_bytes)).decode('ascii')


def graph6_to_adj(g6: str) -> np.ndarray:
    """
    Convert Graph6 string to adjacency matrix.

    Args:
        g6: Graph6 encoded string

    Returns:
        adj: (N, N) adjacency matrix as float64
    """
    data = g6.encode('ascii')
    idx = 0

    # Decode n
    if data[idx] != 126:
        n = data[idx] - 63
        idx += 1
    else:
        idx += 1
        if data[idx] != 126:
            n = ((data[idx] - 63) << 12) | ((data[idx + 1] - 63) << 6) | (data[idx + 2] - 63)
            idx += 3
        else:
            idx += 1
            n = ((data[idx] - 63) << 30) | ((data[idx + 1] - 63) << 24) | \
                ((data[idx + 2] - 63) << 18) | ((data[idx + 3] - 63) << 12) | \
                ((data[idx + 4] - 63) << 6) | (data[idx + 5] - 63)
            idx += 6

    # Decode adjacency bits
    bits = []
    for byte in data[idx:]:
        val = byte - 63
        for i in range(5, -1, -1):
            bits.append((val >> i) & 1)

    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.float64)
    bit_idx = 0
    for j in range(1, n):
        for i in range(j):
            if bit_idx < len(bits) and bits[bit_idx]:
                adj[i, j] = adj[j, i] = 1.0
            bit_idx += 1

    return adj


def graph6_to_torch(g6: str, device: str = 'cpu') -> torch.Tensor:
    """
    Convert Graph6 string to torch adjacency tensor.

    Args:
        g6: Graph6 encoded string
        device: Target device

    Returns:
        adj: (N, N) adjacency tensor as float32
    """
    adj = graph6_to_adj(g6)
    return torch.tensor(adj, dtype=torch.float32, device=device)


def adj_to_edge_list(adj: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert adjacency matrix to edge list.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        edges: (M, 2) array of edge pairs
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()

    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edges.append([i, j])

    return np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def edge_list_to_adj(edges: np.ndarray, n: int) -> np.ndarray:
    """
    Convert edge list to adjacency matrix.

    Args:
        edges: (M, 2) array of edge pairs
        n: Number of nodes

    Returns:
        adj: (N, N) adjacency matrix
    """
    adj = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1.0
    return adj


def count_edges(adj: Union[np.ndarray, torch.Tensor]) -> int:
    """
    Count number of edges in graph.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        m: Number of edges
    """
    if isinstance(adj, torch.Tensor):
        return int(adj.sum().item()) // 2
    else:
        return int(adj.sum()) // 2


def validate_graph6(g6: str) -> bool:
    """
    Validate that a string is valid Graph6 format.

    Args:
        g6: String to validate

    Returns:
        True if valid Graph6, False otherwise
    """
    try:
        adj = graph6_to_adj(g6)
        # Check symmetry
        if not np.allclose(adj, adj.T):
            return False
        # Check no self-loops
        if np.diag(adj).any():
            return False
        return True
    except:
        return False
