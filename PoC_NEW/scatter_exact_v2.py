"""
Deterministic scatter with EXACTLY nd/2 edges, ZERO collisions.

Key insight: use d LAYERS, but each node only creates a directed edge
in layers where i < j (i.e., the target j is LARGER than the source i).
This guarantees: every directed edge i→j has i < j, so no mutual pairs
are possible. After symmetrization, each directed edge = one undirected edge.

With d layers, each node creates ~d/2 directed edges (roughly half the
targets will be > i, half < i). Total: n × d/2 = nd/2 directed edges.
Each becomes one undirected edge. Exactly nd/2.

The "roughly half" isn't exact — some nodes near 0 have most targets > i,
nodes near n-1 have most targets < i. But across all nodes, by equi-
distribution of the QR scatter, exactly half the directed edges satisfy
i < j. So the total is exactly nd/2 (up to O(d) rounding).

Actually, simpler: don't filter by i < j. Instead, use the PAIRING
approach. In each layer k, the QR formula defines a MAP π_k: i → j_k(i).
This map has ~n/2 "forward" edges (i < j_k(i)) and ~n/2 "backward"
edges (i > j_k(i)). Keep ONLY the forward edges. This gives exactly
the number of unique undirected edges without any collisions.

Let's test this.
"""

import numpy as np
from scipy.linalg import eigvalsh


def next_prime(n):
    p = n + 1
    if p <= 2: return 2
    if p % 2 == 0: p += 1
    while True:
        if all(p % f != 0 for f in range(2, int(p**0.5)+1)): return p
        p += 2


def primitive_root(p):
    if p == 2: return 1
    pm1 = p - 1; factors = set(); tmp = pm1; dd = 2
    while dd*dd <= tmp:
        while tmp % dd == 0: factors.add(dd); tmp //= dd
        dd += 1
    if tmp > 1: factors.add(tmp)
    for g in range(2, p):
        if all(pow(g, pm1//f, p) != 1 for f in factors): return g
    return 2


def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


def scatter_forward_only(n, d):
    """
    d layers of QR scatter, but only keep edges where i < j.
    Each layer produces ~n/2 forward edges.
    d layers × n/2 = nd/2 edges. No collisions possible.
    """
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    edge_count = 0

    for k in range(d):
        c = pow(g, k + 1, p)
        for i in range(n):
            t = (i + 1) % p
            if t == 0: t = 1
            j = (t * (t + c)) % p % n
            if j == i: j = (j + 1) % n
            # ONLY keep if i < j (forward edge)
            if i < j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_count += 1

    return adj, edge_count


def scatter_exactly_m(n, d):
    """
    Produce EXACTLY m = nd/2 edges using forward-only QR scatter.

    Uses d layers. If we get more than m, stop early.
    If we get fewer (unlikely), add more layers.
    """
    m = n * d // 2
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    edge_count = 0

    k = 0
    while edge_count < m:
        c = pow(g, k + 1, p)
        for i in range(n):
            t = (i + 1) % p
            if t == 0: t = 1
            j = (t * (t + c)) % p % n
            if j == i: j = (j + 1) % n
            if i < j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_count += 1
                if edge_count >= m:
                    break
        k += 1

    return adj, edge_count, k


def regularize(adj, d):
    """Degree equalization only (no edge count change)."""
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)
    current = int(adj.sum()) // 2
    target = n * d // 2
    assert current == target, f"edges={current}, target={target}"

    for _ in range(n * n):
        over = [i for i in range(n) if degs[i] > d]
        under = [i for i in range(n) if degs[i] < d]
        if not over and not under: break
        if not over or not under: break
        u = min(over, key=lambda i: (-degs[i], i))
        w = min(under, key=lambda i: (degs[i], i))
        neighbors_u = sorted(v for v in range(n) if adj[u, v])
        candidates = sorted(neighbors_u, key=lambda v: (-degs[v], v))
        transferred = False
        for v in candidates:
            if v == w: continue
            if not adj[w, v]:
                adj[u, v] = 0; adj[v, u] = 0
                adj[w, v] = 1; adj[v, w] = 1
                degs[u] -= 1; degs[w] += 1
                transferred = True; break
        if transferred: continue
        if not adj[u, w]:
            adj[u, w] = 1; adj[w, u] = 1; degs[u] += 1; degs[w] += 1
            others = sorted(v for v in range(n) if adj[u, v] and v != w)
            if others:
                bd = max(degs[v] for v in others)
                v = next(nb for nb in others if degs[nb] == bd)
                adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
        else:
            v = max(neighbors_u, key=lambda x: (degs[x], -x))
            adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1; adj[x, w] = 1; degs[w] += 1; degs[x] += 1; break
    return adj


print("=== Approach 1: d layers, forward-only (i < j) ===")
print(f"{'n':>5} {'d':>3} {'target':>7} | {'edges':>6} {'layers':>6} {'Δ':>4} | "
      f"{'λ₂(scat)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
print("-" * 80)

for n in [32, 64, 128, 256]:
    for d in [4, 6, 8]:
        if (n * d) % 2 != 0: continue
        target = n * d // 2

        adj_s, edges, layers_used = scatter_exactly_m(n, d)
        delta = edges - target

        adj_r = regularize(adj_s, d)
        degs = adj_r.sum(axis=1)
        is_reg = np.all(degs == d)

        l2_s = lambda2(adj_s)
        l2_r = lambda2(adj_r)
        ram = ramanujan(d)

        print(f"{n:5d} {d:3d} {target:7d} | {edges:6d} {layers_used:6d} {delta:4d} | "
              f"{l2_s:9.4f} {l2_r:8.4f} {ram:8.4f} {l2_r/ram:6.2f} {'YES' if is_reg else 'NO':>5s}")
    print()
