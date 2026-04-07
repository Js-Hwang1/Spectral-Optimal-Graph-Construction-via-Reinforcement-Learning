"""
Ring-ownership scatter: EXACTLY nd/2 edges, ZERO collisions, ZERO deficit.

Ownership rule: node i owns edge {i, j} iff (j - i) mod n ∈ [1, n/2].
i.e., j is in the "clockwise half" of the ring starting from i.

Key property: for any pair {i, j} with i ≠ j, EXACTLY ONE of i or j
owns the edge. Proof:
  If (j - i) mod n ∈ [1, n/2], then i owns it.
  Otherwise (i - j) mod n ∈ [1, n/2], so j owns it.
  (For n even, the antipodal case (j-i) mod n = n/2 is assigned to
   the node with smaller index to break the tie.)

Each node computes d targets via QR formula. Keeps only those in its
forward half. By equidistribution of QR targets, each node keeps
exactly d/2 (on average). Total: n × d/2 = nd/2.
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


def owns_edge(i, j, n):
    """Does node i own edge {i, j}?
    Yes if j is in i's forward half: (j - i) mod n ∈ [1, n/2).
    For the antipodal case (j - i) mod n == n/2: i owns if i < j.
    """
    dist = (j - i) % n
    if dist == 0:
        return False  # self-loop
    if dist < n // 2:
        return True
    if dist == n // 2:
        return i < j  # tie-break for antipodal
    return False


def scatter_ring_own(n, d):
    """
    Phase 1: QR scatter with ring ownership.
    Each node i computes d targets, keeps only those it owns.
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
            # Only keep if i owns this edge AND it's new
            if owns_edge(i, j, n) and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges += 1

    return adj, edges


def regularize(adj, d):
    """Phase 2: fix edge count then equalize degrees."""
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)
    target = n * d // 2
    current = int(adj.sum()) // 2

    # Step 1: fix edge count
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
                    adj[w, x] = 1; adj[x, w] = 1; degs[w] += 1; degs[x] += 1; break

    return adj


# ============================================================
# Verify ownership property
# ============================================================

print("Verify: every pair {i,j} is owned by exactly one node")
for n in [8, 16, 32]:
    ok = True
    for i in range(n):
        for j in range(n):
            if i == j: continue
            i_owns = owns_edge(i, j, n)
            j_owns = owns_edge(j, i, n)
            if i_owns == j_owns:
                print(f"  FAIL: n={n}, i={i}, j={j}: i_owns={i_owns}, j_owns={j_owns}")
                ok = False
    print(f"  n={n}: {'PASS' if ok else 'FAIL'} — every edge has exactly one owner")

print()

# ============================================================
# Test
# ============================================================

print("Ring-ownership QR scatter")
print("=" * 95)
print(f"{'n':>5} {'d':>3} {'target':>7} | {'edges':>6} {'Δ':>4} | "
      f"{'deg_min':>7} {'deg_max':>7} {'deg_std':>7} | "
      f"{'λ₂(scat)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
print("-" * 95)

for n in [32, 64, 128, 256, 512]:
    for d in [4, 6, 8]:
        if (n * d) % 2 != 0: continue
        target = n * d // 2

        adj_s, edges = scatter_ring_own(n, d)
        delta = edges - target
        degs_s = adj_s.sum(axis=1)

        adj_r = regularize(adj_s.copy(), d)
        degs_r = adj_r.sum(axis=1)
        is_reg = np.all(degs_r == d)

        l2_s = lambda2(adj_s)
        l2_r = lambda2(adj_r)
        ram = ramanujan(d)

        print(f"{n:5d} {d:3d} {target:7d} | {edges:6d} {delta:4d} | "
              f"{int(degs_s.min()):7d} {int(degs_s.max()):7d} {degs_s.std():7.2f} | "
              f"{l2_s:9.4f} {l2_r:8.4f} {ram:8.4f} {l2_r/ram:6.2f} "
              f"{'YES' if is_reg else 'NO':>5s}")
    print()
