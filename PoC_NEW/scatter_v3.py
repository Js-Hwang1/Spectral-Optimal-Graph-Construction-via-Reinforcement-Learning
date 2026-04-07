"""
Clean rethink. No layers. No bullshit.

Each node i independently picks d/2 neighbors via:
    j_k = f(i, k, n)  for k = 0, ..., d/2-1

Total: n * d/2 = nd/2 directed edges.
If zero mutual collisions: nd/2 undirected edges = EXACT target.

The formula f must satisfy:
  1. j != i (no self-loops)
  2. j_0, j_1, ..., j_{d/2-1} are distinct (no multi-edges from same node)
  3. If f(i, k, n) = j, then f(j, k', n) != i for all k' (no mutuals)

Condition 3 is the hard one. Let's test various formulas.
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


def test_scatter(name, target_fn, n, d):
    """
    Test a scatter function.
    target_fn(i, k, n, p, g) -> j   (the k-th target of node i)

    Returns (adj, stats_dict)
    """
    p = next_prime(n)
    g = primitive_root(p)
    half_d = d // 2

    # Each node picks d/2 targets
    directed = set()
    for i in range(n):
        for k in range(half_d):
            j = target_fn(i, k, n, p, g)
            assert 0 <= j < n, f"j={j} out of range"
            if j != i:
                directed.add((i, j))

    # Check mutuals
    mutuals = 0
    for (a, b) in directed:
        if (b, a) in directed:
            mutuals += 1
    mutuals //= 2  # each mutual counted twice

    # Undirected edges
    undirected = set()
    for (a, b) in directed:
        undirected.add((min(a, b), max(a, b)))

    # Build adjacency
    adj = np.zeros((n, n), dtype=np.int8)
    for (a, b) in undirected:
        adj[a, b] = 1; adj[b, a] = 1

    target_m = n * d // 2
    return adj, {
        'directed': len(directed),
        'undirected': len(undirected),
        'mutuals': mutuals,
        'target': target_m,
        'delta': len(undirected) - target_m,
    }


# ============================================================
# Scatter formulas
# ============================================================

def qr_scatter(i, k, n, p, g):
    """Original QR: j = t*(t+c_k) mod p mod n"""
    c = pow(g, k + 1, p)
    t = (i + 1) % p
    if t == 0: t = 1
    j = (t * (t + c)) % p % n
    if j == i: j = (j + 1) % n
    return j


def power_perm(i, k, n, p, g):
    """Power permutation: j = (i+1)^e_k mod p mod n (bijection on Z_p*)"""
    pm1 = p - 1
    e = int(((k + 1) * ((1+5**0.5)/2) * pm1) % pm1) + 2
    while np.gcd(e, pm1) != 1:
        e += 1
    j = pow(i + 1, e, p) % n
    if j == i: j = (j + 1) % n
    return j


def affine_scatter(i, k, n, p, g):
    """Affine: j = (a_k * i + b_k) mod p mod n (bijection)"""
    a = pow(g, k + 1, p)  # multiplier (nonzero)
    b = pow(g, k + 7, p)  # offset
    j = (a * (i + 1) + b) % p % n
    if j == i: j = (j + 1) % n
    return j


def mixed_scatter(i, k, n, p, g):
    """QR with node-dependent offset: j = t*(t + c_k + i*g) mod p mod n"""
    c = pow(g, k + 1, p)
    t = (i + 1) % p
    if t == 0: t = 1
    j = (t * (t + c)) % p % n
    if j == i: j = (j + 1) % n
    return j


# ============================================================
# Test
# ============================================================

formulas = [
    ("QR scatter", qr_scatter),
    ("Power perm", power_perm),
    ("Affine", affine_scatter),
]

print(f"{'formula':>12} {'n':>5} {'d':>3} | {'directed':>8} {'undir':>6} {'mutual':>7} {'target':>7} {'Δ':>4}")
print("-" * 65)

for name, fn in formulas:
    for n in [32, 64, 128, 256]:
        for d in [4, 6]:
            if (n * d) % 2 != 0: continue
            adj, stats = test_scatter(name, fn, n, d)
            print(f"{name:>12} {n:5d} {d:3d} | {stats['directed']:8d} {stats['undirected']:6d} "
                  f"{stats['mutuals']:7d} {stats['target']:7d} {stats['delta']:4d}")
    print()

# Now test with regularization for the best formula
print("\n=== With regularization ===")
print(f"{'formula':>12} {'n':>5} {'d':>3} | {'Δ_before':>8} {'Δ_after':>8} | "
      f"{'λ₂':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
print("-" * 75)

from random_init_reg import regularize as reg_exact

for name, fn in formulas:
    for n in [64, 128, 256]:
        for d in [4, 6]:
            if (n * d) % 2 != 0: continue
            adj, stats = test_scatter(name, fn, n, d)
            m = stats['target']
            e = stats['undirected']
            delta_before = stats['delta']

            # If deficit, add edges; if surplus, remove
            degs = adj.sum(axis=1).astype(int)
            current = int(adj.sum()) // 2
            while current < m:
                min_deg = int(degs.min())
                u = next(i for i in range(n) if degs[i] == min_deg)
                for v in range(n):
                    if v != u and not adj[u, v] and degs[v] == min_deg:
                        adj[u, v] = 1; adj[v, u] = 1
                        degs[u] += 1; degs[v] += 1; current += 1; break
                else: break
            while current > m:
                max_deg = int(degs.max())
                u = next(i for i in range(n) if degs[i] == max_deg)
                nbs = [v for v in range(n) if adj[u, v]]
                v = max(nbs, key=lambda x: degs[x])
                adj[u, v] = 0; adj[v, u] = 0
                degs[u] -= 1; degs[v] -= 1; current -= 1

            delta_after = int(adj.sum()) // 2 - m
            adj_r = reg_exact(adj, d)
            is_reg = np.all(adj_r.sum(axis=1) == d)
            l2 = lambda2(adj_r)
            ram = ramanujan(d)

            print(f"{name:>12} {n:5d} {d:3d} | {delta_before:8d} {delta_after:8d} | "
                  f"{l2:8.4f} {ram:8.4f} {l2/ram:6.2f} {'YES' if is_reg else 'NO':>5s}")
    print()
