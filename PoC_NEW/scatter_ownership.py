"""
Ownership-based scatter: EXACTLY nd/2 edges, ZERO collisions.

Each node i computes d targets j_0, ..., j_{d-1} via QR formula.
But node i only CLAIMS edge {i, j_k} if i < j_k (i is the "owner").

Since every edge {a, b} with a < b can only be claimed by node a,
there are ZERO collisions — no two nodes claim the same edge.

Each node claims ~d/2 edges (roughly half its targets are > i).
Total: n × d/2 = nd/2 edges. Exact by equidistribution.

The degree distribution will be imbalanced (not d-regular) —
that's exactly what we want for Phase 2.
"""

import numpy as np
from scipy.linalg import eigvalsh


def next_prime(n):
    p = n + 1
    if p <= 2: return 2
    if p % 2 == 0: p += 1
    while True:
        if all(p % f != 0 for f in range(2, int(p**0.5) + 1)): return p
        p += 2


def primitive_root(p):
    if p == 2: return 1
    pm1 = p - 1; factors = set(); tmp = pm1; dd = 2
    while dd * dd <= tmp:
        while tmp % dd == 0: factors.add(dd); tmp //= dd
        dd += 1
    if tmp > 1: factors.add(tmp)
    for g in range(2, p):
        if all(pow(g, pm1 // f, p) != 1 for f in factors): return g
    return 2


def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


def scatter_ownership(n, d):
    """
    Phase 1: QR scatter with ownership rule.

    Each node i computes d targets. Only keeps edges where i < j.
    Zero collisions guaranteed. Edge count ≈ nd/2.
    """
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    edges = 0

    for i in range(n):
        for k in range(d):
            c = pow(g, k + 1, p)
            t = (i + 1) % p
            if t == 0: t = 1
            j = (t * (t + c)) % p % n
            if j == i: j = (j + 1) % n
            # Ownership: only add if i < j AND edge doesn't exist yet
            if i < j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges += 1

    return adj, edges


def regularize(adj, d):
    """Phase 2: edge-count preserving degree equalization."""
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)
    target = n * d // 2
    current = int(adj.sum()) // 2

    # Step 1: fix edge count if needed
    while current > target:
        max_deg = int(degs.max())
        u = next(i for i in range(n) if degs[i] == max_deg)
        nbs = sorted(v for v in range(n) if adj[u, v])
        if not nbs: break
        bd = max(degs[v] for v in nbs)
        v = next(nb for nb in nbs if degs[nb] == bd)
        adj[u, v] = 0; adj[v, u] = 0
        degs[u] -= 1; degs[v] -= 1; current -= 1

    while current < target:
        min_deg = int(degs.min())
        u = next(i for i in range(n) if degs[i] == min_deg)
        best_v = -1; best_d = n + 1
        for v in range(n):
            if v != u and not adj[u, v] and degs[v] < best_d:
                best_d = degs[v]; best_v = v
        if best_v < 0: break
        adj[u, best_v] = 1; adj[best_v, u] = 1
        degs[u] += 1; degs[best_v] += 1; current += 1

    # Step 2: degree equalization via swaps
    for _ in range(n * n):
        over = [i for i in range(n) if degs[i] > d]
        under = [i for i in range(n) if degs[i] < d]
        if not over and not under: break
        if not over or not under: break
        u = min(over, key=lambda i: (-degs[i], i))
        w = min(under, key=lambda i: (degs[i], i))
        nbs_u = sorted(v for v in range(n) if adj[u, v])
        cands = sorted(nbs_u, key=lambda v: (-degs[v], v))
        done = False
        for v in cands:
            if v == w: continue
            if not adj[w, v]:
                adj[u, v] = 0; adj[v, u] = 0
                adj[w, v] = 1; adj[v, w] = 1
                degs[u] -= 1; degs[w] += 1
                done = True; break
        if done: continue
        if not adj[u, w]:
            adj[u, w] = 1; adj[w, u] = 1; degs[u] += 1; degs[w] += 1
            others = sorted(v for v in range(n) if adj[u, v] and v != w)
            if others:
                bd = max(degs[v] for v in others)
                v = next(nb for nb in others if degs[nb] == bd)
                adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
        else:
            v = max(nbs_u, key=lambda x: (degs[x], -x))
            adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1; adj[x, w] = 1
                    degs[w] += 1; degs[x] += 1; break

    return adj


# ============================================================
# Test
# ============================================================

print("Ownership-based QR scatter: node i keeps edge {i,j} only if i < j")
print("=" * 80)
print(f"{'n':>5} {'d':>3} {'target':>7} | {'edges':>6} {'Δ':>4} | "
      f"{'deg_min':>7} {'deg_max':>7} {'deg_std':>7} | "
      f"{'λ₂(scat)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
print("-" * 95)

for n in [32, 64, 128, 256, 512]:
    for d in [4, 6, 8]:
        if (n * d) % 2 != 0: continue
        target = n * d // 2

        adj_s, edges = scatter_ownership(n, d)
        delta = edges - target
        degs_s = adj_s.sum(axis=1)

        adj_r = regularize(adj_s.copy(), d)
        degs_r = adj_r.sum(axis=1)
        is_reg = np.all(degs_r == d)
        edges_r = int(adj_r.sum()) // 2

        l2_s = lambda2(adj_s)
        l2_r = lambda2(adj_r)
        ram = ramanujan(d)

        print(f"{n:5d} {d:3d} {target:7d} | {edges:6d} {delta:4d} | "
              f"{int(degs_s.min()):7d} {int(degs_s.max()):7d} {degs_s.std():7.2f} | "
              f"{l2_s:9.4f} {l2_r:8.4f} {ram:8.4f} {l2_r/ram:6.2f} "
              f"{'YES' if is_reg else 'NO':>5s}")
    print()
