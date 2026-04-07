"""
Edge-centric QR scatter v2: better scrambling.

Fix: use the QR polynomial to determine BOTH the source and target
for each edge slot, not just the target.

For edge slot e:
  Use two independent QR evaluations:
    s = f_1(e) mod n   (source node — scrambled, not sequential)
    t = f_2(e) mod n   (target node — scrambled independently)
  where f_1 and f_2 use different coefficients from the primitive root.

This ensures both source and target are equidistributed and
decorrelated from the slot index.
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


def scatter_v2(n, d):
    """
    Edge-centric scatter v2.

    For edge slot e = 0, 1, ...:
      t = (e + 1) mod p; if t == 0: t = 1
      i = t^2 mod p mod n                     (source: QR of e)
      j = t * (t + c) mod p mod n             (target: different QR of e)
      c = g^(floor(e * g / m) + 1) mod p      (coefficient varies smoothly)

    Both i and j are nonlinear functions of e.
    """
    m = n * d // 2
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    e = 0

    while count < m:
        t = (e + 1) % p
        if t == 0: t = 1

        # Source: one QR polynomial
        i = (t * t) % p % n

        # Target: different QR polynomial with varying coefficient
        c = pow(g, (e % (p - 1)) + 1, p)
        j = (t * (t + c)) % p % n

        # Self-loop avoidance
        if j == i:
            j = (j + 1) % n

        # Accept if new edge
        if adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1

        e += 1
        if e > 4 * m:
            print(f"WARNING: e={e}, count={count}/{m}")
            break

    return adj, count, e


def scatter_v3(n, d):
    """
    Edge-centric scatter v3: use primitive root powers as edge enumeration.

    Idea: g^e mod p cycles through all of F_p* (full period p-1).
    Use this to generate both source and target:
      val = g^e mod p
      i = val mod n
      j = (val * (val + g)) mod p mod n   (QR applied to val)

    Since g^e visits every element of F_p* exactly once in p-1 steps,
    the (i, j) pairs are maximally spread.
    """
    m = n * d // 2
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    count = 0

    for e in range(p - 1):
        val = pow(g, e + 1, p)  # g^(e+1) mod p, cycles through F_p*

        i = val % n
        j = (val * (val + g)) % p % n

        if j == i:
            j = (j + 1) % n

        if adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1

        if count >= m:
            break

    return adj, count, e + 1


def scatter_v4(n, d):
    """
    Edge-centric scatter v4: Lehmer-style full-period enumeration.

    Use g^e mod p to enumerate edge slots.
    For each slot, derive source and target from INDEPENDENT halves:
      val = g^e mod p
      i = val mod n
      j = (g^(e + p//2)) mod p mod n    (phase-shifted by half period)

    Since g^e and g^(e + p//2) are decorrelated (half-period apart),
    i and j are approximately independent.
    """
    m = n * d // 2
    p = next_prime(n)
    g = primitive_root(p)
    half_p = (p - 1) // 2

    adj = np.zeros((n, n), dtype=np.int8)
    count = 0

    for e in range(p - 1):
        val1 = pow(g, e + 1, p)
        val2 = pow(g, e + 1 + half_p, p)

        i = val1 % n
        j = val2 % n

        if j == i:
            j = (j + 1) % n

        if adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1

        if count >= m:
            break

    return adj, count, e + 1


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
# Test all variants
# ============================================================

variants = [
    ("v2: dual QR", scatter_v2),
    ("v3: g^e QR", scatter_v3),
    ("v4: half-period", scatter_v4),
]

for vname, vfn in variants:
    print(f"\n=== {vname} ===")
    print(f"{'n':>5} {'d':>3} {'m':>7} | {'edges':>6} {'iters':>6} | "
          f"{'deg_std':>7} | {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
    print("-" * 75)

    for n in [32, 64, 128, 256, 512]:
        for d in [4, 6, 8]:
            if (n * d) % 2 != 0: continue
            m = n * d // 2

            adj_s, count, iters = vfn(n, d)
            if count < m:
                print(f"{n:5d} {d:3d} {m:7d} | {count:6d} {iters:6d} | INSUFFICIENT EDGES")
                continue

            degs_s = adj_s.sum(axis=1)
            adj_r = regularize(adj_s.copy(), d)
            degs_r = adj_r.sum(axis=1)
            is_reg = np.all(degs_r == d)
            l2_r = lambda2(adj_r)
            ram = ramanujan(d)

            print(f"{n:5d} {d:3d} {m:7d} | {count:6d} {iters:6d} | "
                  f"{degs_s.std():7.2f} | "
                  f"{l2_r:8.4f} {ram:8.4f} {l2_r/ram:6.2f} {'YES' if is_reg else 'NO':>5s}")
    print()
