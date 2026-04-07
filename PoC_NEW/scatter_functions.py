"""
Shootout: find a deterministic edge-generation function that matches
or beats true random at ALL scales.

All use the same framework:
  - Edge-centric: iterate e = 0, 1, ..., stop at exactly m = nd/2
  - For each e, compute (i, j) via the formula
  - Skip duplicates and self-loops
  - Then regularize to d-regular

Functions to test:
  1. QR (degree 2): t*(t+c) mod p — current, has correlations
  2. Cubic: t³ + a*t + b mod p — degree 3, less correlation
  3. Power residue: t^e mod p with gcd(e, p-1)=1 — bijection, full scatter
  4. Inversive: (a/t + b) mod p — proven low autocorrelation (ICG)
  5. Lehmer: g^(hash(e)) mod p — exponential scatter
  6. Feistel: 2-round Feistel on e, then extract (i, j)
  7. Double-hash: two independent hashes for i and j
"""

import numpy as np
from scipy.linalg import eigvalsh
from random_init_reg import random_init, regularize


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


def edge_centric_build(n, d, edge_fn):
    """
    Generic edge-centric builder.
    edge_fn(e, n, p, g) -> (i, j)
    Stops at exactly m = nd/2 edges.
    """
    m = n * d // 2
    p = next_prime(n)
    g = primitive_root(p)

    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    e = 0
    while count < m:
        i, j = edge_fn(e, n, p, g)
        i = i % n
        j = j % n
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1
        e += 1
        if e > 6 * m:
            break
    return adj, count, e


# ============================================================
# Edge functions
# ============================================================

def fn_qr(e, n, p, g):
    """Degree-2 QR: current approach."""
    t = (e + 1) % p
    if t == 0: t = 1
    c = pow(g, (e % (p - 1)) + 1, p)
    i = (t * t) % p % n
    j = (t * (t + c)) % p % n
    return i, j


def fn_cubic(e, n, p, g):
    """Degree-3 cubic polynomial."""
    t = (e + 1) % p
    if t == 0: t = 1
    a = pow(g, 3, p)
    b = pow(g, 7, p)
    c = pow(g, (e % (p - 1)) + 1, p)
    i = (t * t * t + a * t) % p % n
    j = (t * t * t + c * t + b) % p % n
    return i, j


def fn_power(e, n, p, g):
    """Power residue: t^e1, t^e2 with coprime exponents."""
    t = (e + 1) % p
    if t == 0: t = 1
    pm1 = p - 1
    e1 = 3  # coprime to p-1 for most p
    while np.gcd(e1, pm1) != 1: e1 += 1
    e2 = e1 + 2
    while np.gcd(e2, pm1) != 1: e2 += 1
    layer = (e // n) + 1
    i = pow(t, e1 * layer, p) % n
    j = pow(t, e2 * layer, p) % n
    return i, j


def fn_inversive(e, n, p, g):
    """Inversive congruential: a/t + b mod p. Proven low autocorrelation."""
    t = (e + 1) % p
    if t == 0: t = 1
    a = pow(g, 2, p)
    b1 = pow(g, (e // n) + 3, p)
    b2 = pow(g, (e // n) + 7, p)
    inv_t = pow(t, p - 2, p)  # Fermat inverse
    i = (a * inv_t + b1) % p % n
    j = (a * inv_t * inv_t + b2) % p % n  # quadratic in 1/t
    return i, j


def fn_lehmer(e, n, p, g):
    """Lehmer/exponential: g^(e-derived) mod p for both i and j."""
    # Use two decorrelated exponentials
    i = pow(g, 2 * e + 1, p) % n
    j = pow(g, 2 * e + 2, p) % n
    return i, j


def fn_composite(e, n, p, g):
    """Composition: power perm + inversive."""
    t = (e + 1) % p
    if t == 0: t = 1
    pm1 = p - 1
    exp = 3
    while np.gcd(exp, pm1) != 1: exp += 1
    layer = (e // n) + 1
    # Step 1: power permutation
    x = pow(t, exp * layer, p)
    # Step 2: inversive on x
    inv_x = pow(x if x != 0 else 1, p - 2, p)
    c = pow(g, layer + 5, p)
    i = x % n
    j = (pow(g, 2, p) * inv_x + c) % p % n
    return i, j


def fn_double_lehmer(e, n, p, g):
    """Two completely independent Lehmer sequences for i and j."""
    # Use two different primitive roots (or same root, widely spaced)
    # g^e and g^(e + (p-1)/2) are decorrelated
    half = (p - 1) // 2
    i = pow(g, e + 1, p) % n
    j = pow(g, e + 1 + half, p) % n
    return i, j


def fn_xorshift(e, n, p, g):
    """Xorshift-like deterministic hash for maximum decorrelation."""
    # Mix e through bit operations, then mod n
    x = e + 1
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    i = x % n
    # Different hash for j
    y = e + 0x9e3779b9
    y = ((y ^ (y >> 16)) * 0x85ebca6b) & 0xFFFFFFFF
    y = ((y ^ (y >> 16)) * 0xc2b2ae35) & 0xFFFFFFFF
    y = y ^ (y >> 16)
    j = y % n
    return i, j


# ============================================================
# Test
# ============================================================

functions = [
    ("QR (deg 2)", fn_qr),
    ("Cubic (deg 3)", fn_cubic),
    ("Power perm", fn_power),
    ("Inversive", fn_inversive),
    ("Lehmer", fn_lehmer),
    ("Composite", fn_composite),
    ("Dbl Lehmer", fn_double_lehmer),
    ("Xorshift", fn_xorshift),
]

print("Scramble function shootout: which matches true random?")
print("All use edge-centric build + same regularizer.")
print("=" * 90)

for n in [64, 128, 256, 512]:
    d = 4
    if (n * d) % 2 != 0: continue
    m = n * d // 2
    ram = ramanujan(d)

    # Random baseline (5 seeds)
    rd_l2s = []
    for seed in range(5):
        adj = random_init(n, d, seed=seed)
        adj = regularize(adj, d)
        rd_l2s.append(lambda2(adj))
    rd_mean = np.mean(rd_l2s)

    print(f"\nn={n}, d={d}, m={m}, Ram={ram:.4f}, Random+REG={rd_mean:.4f} ({rd_mean/ram:.2f}x)")
    print(f"{'Function':>14} | {'edges':>5} {'iters':>5} {'ovhd':>5} | "
          f"{'λ₂':>8} {'ratio':>6} {'vs_rand':>8}")
    print("-" * 65)

    for fname, ffn in functions:
        adj, count, iters = edge_centric_build(n, d, ffn)
        if count < m:
            print(f"{fname:>14} | {count:5d} {iters:5d}       | INSUFFICIENT")
            continue
        adj_r = regularize(adj, d)
        l2 = lambda2(adj_r)
        overhead = iters / m
        vs_rand = l2 / rd_mean
        marker = "***" if vs_rand >= 0.99 else "   "
        print(f"{fname:>14} | {count:5d} {iters:5d} {overhead:5.2f} | "
              f"{l2:8.4f} {l2/ram:6.2f}x {vs_rand:8.2f}x {marker}")
