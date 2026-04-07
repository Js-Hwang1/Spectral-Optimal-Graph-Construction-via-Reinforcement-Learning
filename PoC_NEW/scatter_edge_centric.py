"""
Edge-centric scatter: iterate over m = nd/2 edge slots.
Each slot e maps to a unique edge {i, j} via analytical formula.
Exactly m edges. Zero surplus. Zero deficit.

Edge slot e ∈ [0, m-1] maps to (i, j) via:
  t = (e + 1) mod p          (map slot to F_p element)
  c = g^(e // n + 1) mod p   (layer-like coefficient derived from slot)
  raw = t * (t + c) mod p
  i = e mod n                (source node cycles through [0, n-1])
  j = raw mod n              (target from QR formula)

But this doesn't guarantee uniqueness. Better approach:

Use the QR polynomial to generate a PERMUTATION on edge indices.
Map each edge slot e to a unique pair (i, j) from the set of all
possible edges, using the QR formula as a hash.

Actually, simplest correct approach:
  - There are n*(n-1)/2 possible edges.
  - We want to select m = nd/2 of them deterministically.
  - Use the QR formula to generate a permutation of [0, n*(n-1)/2 - 1],
    take the first m elements, decode each to an edge {i, j}.

But n*(n-1)/2 can be huge. Instead:

Approach: iterate e = 0, 1, ..., and for each e compute a candidate
edge via the QR formula. Skip duplicates. Stop at exactly m.
Since the QR formula is equidistributed, we expect ~m iterations
with O(1) retries for duplicates at low density (m << n²/2).
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


def scatter_edge_centric(n, d):
    """
    Edge-centric QR scatter. Exactly m = nd/2 edges.

    For edge slot e = 0, 1, 2, ...:
      Map e to a candidate edge (i, j) via QR polynomial.
      If edge is new and not a self-loop, accept it.
      Stop when we have exactly m edges.

    The mapping:
      p = next_prime(n)
      g = primitive root mod p
      t = (e + 1) mod p;  if t == 0: t = 1
      c = g^(floor(e / n) + 1) mod p   (changes every n slots)
      val = t * (t + c) mod p
      i = e mod n
      j = val mod n
      if i == j: j = (j + 1) mod n

    This cycles i through [0, n-1] via e mod n, and uses the QR
    polynomial to scatter j. The coefficient c changes every n edges,
    providing fresh randomness across "layers" of slots.
    """
    m = n * d // 2
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    e = 0

    while count < m:
        # Source: cycle through nodes
        i = e % n

        # Coefficient: changes every n slots (like layers)
        layer = e // n
        c = pow(g, layer + 1, p)

        # Target: QR polynomial
        t = (e + 1) % p
        if t == 0: t = 1
        j = (t * (t + c)) % p % n

        # Self-loop avoidance
        if j == i:
            j = (j + 1) % n

        # Only accept if edge is new
        if adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1

        e += 1

        # Safety: shouldn't need more than 2m iterations
        if e > 4 * m:
            print(f"WARNING: too many iterations e={e}, count={count}/{m}")
            break

    return adj, count, e


def regularize(adj, d):
    """Phase 2: edge-count preserving degree equalization."""
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)
    target = n * d // 2
    current = int(adj.sum()) // 2
    assert current == target, f"edges={current}, need={target}"

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
# Test
# ============================================================

print("Edge-centric QR scatter: exactly m = nd/2 edges")
print("=" * 100)
print(f"{'n':>5} {'d':>3} {'m':>7} | {'edges':>6} {'iters':>6} {'ratio':>6} | "
      f"{'deg_min':>7} {'deg_max':>7} {'deg_std':>7} | "
      f"{'λ₂(scat)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
print("-" * 100)

for n in [32, 64, 128, 256, 512]:
    for d in [4, 6, 8]:
        if (n * d) % 2 != 0: continue
        m = n * d // 2

        adj_s, count, iters = scatter_edge_centric(n, d)
        degs_s = adj_s.sum(axis=1)

        adj_r = regularize(adj_s.copy(), d)
        degs_r = adj_r.sum(axis=1)
        is_reg = np.all(degs_r == d)

        l2_s = lambda2(adj_s)
        l2_r = lambda2(adj_r)
        ram = ramanujan(d)

        print(f"{n:5d} {d:3d} {m:7d} | {count:6d} {iters:6d} {iters/m:6.2f} | "
              f"{int(degs_s.min()):7d} {int(degs_s.max()):7d} {degs_s.std():7.2f} | "
              f"{l2_s:9.4f} {l2_r:8.4f} {ram:8.4f} {l2_r/ram:6.2f} "
              f"{'YES' if is_reg else 'NO':>5s}")
    print()
