"""
QRS-DR: Quadratic Residue Scatter + Degree Regularization.
CLEAN implementation. No legacy cruft.

Phase 1: QR Scatter
  - d/2 layers (each layer produces ~n undirected edges)
  - d/2 layers × n edges ≈ nd/2 = target edge count
  - Formula: j = t*(t + g^(k+1)) mod p mod n, t = (i+1) mod p

Phase 2: Degree Regularization
  - Deterministic, canonical tie-breaking (lowest index)
  - Step 1: adjust edge count to exactly nd/2
  - Step 2: equalize all degrees to exactly d
"""

import numpy as np
from scipy.linalg import eigvalsh


# ============================================================
# Number theory
# ============================================================

def next_prime(n):
    """Smallest prime strictly greater than n."""
    p = n + 1
    if p <= 2: return 2
    if p % 2 == 0: p += 1
    while True:
        if all(p % f != 0 for f in range(2, int(p**0.5) + 1)):
            return p
        p += 2


def primitive_root(p):
    """Smallest primitive root modulo p."""
    if p == 2: return 1
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


# ============================================================
# Phase 1: QR Scatter
# ============================================================

def qr_scatter(n, d):
    """
    Phase 1: Quadratic Residue Scatter.

    Uses d/2 layers. Each layer k:
      c_k = g^(k+1) mod p
      for each node i: j = (i+1)*((i+1) + c_k) mod p mod n

    Each layer produces ~n undirected edges.
    d/2 layers → ~nd/2 edges ≈ target for d-regular.
    """
    assert d >= 3 and n >= 4 and (n * d) % 2 == 0

    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n)
    g = primitive_root(p)

    num_layers = d // 2  # KEY: d/2 layers, not d

    for k in range(num_layers):
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


# ============================================================
# Phase 2: Degree Regularization
# ============================================================

def regularize(adj, d):
    """
    Phase 2: Deterministic degree regularization.
    All ties broken by lowest node index.
    """
    n = adj.shape[0]
    adj = adj.copy()
    np.fill_diagonal(adj, 0)
    adj = np.clip(adj, 0, 1)

    target_edges = n * d // 2
    degs = adj.sum(axis=1).astype(int)
    current_edges = int(adj.sum()) // 2

    # Step 1: Fix total edge count
    while current_edges > target_edges:
        max_deg = int(degs.max())
        u = next(i for i in range(n) if degs[i] == max_deg)
        neighbors = sorted(v for v in range(n) if adj[u, v])
        if not neighbors: break
        best_deg = max(degs[v] for v in neighbors)
        v = next(nb for nb in neighbors if degs[nb] == best_deg)
        adj[u, v] = 0; adj[v, u] = 0
        degs[u] -= 1; degs[v] -= 1; current_edges -= 1

    while current_edges < target_edges:
        min_deg = int(degs.min())
        u = next(i for i in range(n) if degs[i] == min_deg)
        best_v = -1; best_deg = n + 1
        for v in range(n):
            if v != u and not adj[u, v] and degs[v] < best_deg:
                best_deg = degs[v]; best_v = v
        if best_v < 0: break
        adj[u, best_v] = 1; adj[best_v, u] = 1
        degs[u] += 1; degs[best_v] += 1; current_edges += 1

    # Step 2: Equalize degrees via edge swaps
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
            adj[u, w] = 1; adj[w, u] = 1
            degs[u] += 1; degs[w] += 1
            others = sorted(v for v in range(n) if adj[u, v] and v != w)
            if others:
                best_deg = max(degs[v] for v in others)
                v = next(nb for nb in others if degs[nb] == best_deg)
                adj[u, v] = 0; adj[v, u] = 0
                degs[u] -= 1; degs[v] -= 1
        else:
            best_deg = max(degs[v] for v in neighbors_u)
            v = next(nb for nb in neighbors_u if degs[nb] == best_deg)
            adj[u, v] = 0; adj[v, u] = 0
            degs[u] -= 1; degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1; adj[x, w] = 1
                    degs[w] += 1; degs[x] += 1; break

    return adj


# ============================================================
# Build
# ============================================================

def build(n, d):
    """Build QRS-DR graph: scatter + regularize."""
    return regularize(qr_scatter(n, d), d)


# ============================================================
# Analysis
# ============================================================

def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print(f"{'n':>5} {'d':>3} | {'layers':>6} {'scat_e':>7} {'target':>7} {'Δ':>5} | "
          f"{'λ₂(scat)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
    print("-" * 85)

    for n in [32, 64, 128, 256, 512]:
        for d in [4, 6, 8]:
            if (n * d) % 2 != 0: continue

            adj_s = qr_scatter(n, d)
            edges_s = int(adj_s.sum()) // 2
            target = n * d // 2
            delta = edges_s - target

            adj_r = build(n, d)
            degs = adj_r.sum(axis=1)
            is_reg = np.all(degs == d)
            l2_s = lambda2(adj_s)
            l2_r = lambda2(adj_r)
            ram = ramanujan(d)
            ratio = l2_r / ram

            print(f"{n:5d} {d:3d} | {d//2:6d} {edges_s:7d} {target:7d} {delta:5d} | "
                  f"{l2_s:9.4f} {l2_r:8.4f} {ram:8.4f} {ratio:6.2f} "
                  f"{'YES' if is_reg else 'NO':>5s}")
        print()
