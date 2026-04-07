"""
Native Z_n scrambles — no bit tricks, pure number theory.

All methods work in Z_p (p = next prime >= n+1) then project to [0,n).
The key: use NONLINEAR maps that are proven low-correlation in Z_p.

Methods:
1. ICG+: Inversive congruential with quadratic perturbation
2. Power permutation: i^e mod p where gcd(e, p-1)=1 (proven bijection)
3. Quadratic residue scatter: i*(i+c_k) mod p (2x edges, let regularizer pick)
4. Composite: layer-dependent mix of ICG + power + linear
5. Multi-pass: apply 2 different maps in sequence (composition = better mixing)
"""

import numpy as np
from scipy.linalg import eigvalsh

phi = (1 + np.sqrt(5)) / 2


def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


def rd_mean(n, d, seeds=20):
    rng = np.random.RandomState(42)
    vals = []
    for _ in range(seeds):
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        rng.shuffle(stubs)
        adj = np.zeros((n, n), dtype=np.int8)
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or adj[u, v]:
                ok = False
                break
            adj[u, v] = 1
            adj[v, u] = 1
        if ok:
            vals.append(lambda2(adj))
    return np.mean(vals) if vals else 0


def next_prime(n):
    if n <= 2:
        return 2
    p = n if n % 2 == 1 else n + 1
    while True:
        if all(p % f != 0 for f in range(2, int(p**0.5) + 1)):
            return p
        p += 2


def primitive_root(p):
    if p == 2:
        return 1
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


def mod_inverse(a, p):
    return pow(a, p - 2, p)


# ====================================================================
# Degree regularizer (edge-swap, guaranteed d-regular)
# ====================================================================

def degree_regularize(adj, d):
    n = adj.shape[0]
    adj = np.clip(adj, 0, 1)
    np.fill_diagonal(adj, 0)

    target_edges = n * d // 2
    degs = adj.sum(axis=1).astype(int)

    # Fix total edge count
    current_edges = int(adj.sum()) // 2
    while current_edges > target_edges:
        u = np.argmax(degs)
        neighbors = np.where(adj[u] == 1)[0]
        if len(neighbors) == 0:
            break
        v = neighbors[np.argmax(degs[neighbors])]
        adj[u, v] = 0
        adj[v, u] = 0
        degs[u] -= 1
        degs[v] -= 1
        current_edges -= 1

    while current_edges < target_edges:
        u = np.argmin(degs)
        best_v = -1
        best_deg = n + 1
        for v in range(n):
            if v != u and not adj[u, v] and degs[v] < best_deg:
                best_deg = degs[v]
                best_v = v
        if best_v < 0:
            break
        adj[u, best_v] = 1
        adj[best_v, u] = 1
        degs[u] += 1
        degs[best_v] += 1
        current_edges += 1

    # Edge-swap to equalize degrees
    for _ in range(n * n):
        over = np.where(degs > d)[0]
        under = np.where(degs < d)[0]
        if len(over) == 0 and len(under) == 0:
            break

        u = over[np.argmax(degs[over])]
        w = under[np.argmin(degs[under])]

        neighbors_u = np.where(adj[u] == 1)[0]
        transferred = False
        candidates = sorted(neighbors_u, key=lambda v: -degs[v])
        for v in candidates:
            if v == w:
                continue
            if not adj[w, v]:
                adj[u, v] = 0
                adj[v, u] = 0
                adj[w, v] = 1
                adj[v, w] = 1
                degs[u] -= 1
                degs[w] += 1
                transferred = True
                break

        if not transferred:
            if not adj[u, w]:
                adj[u, w] = 1
                adj[w, u] = 1
                degs[u] += 1
                degs[w] += 1
                neighbors_u = np.where(adj[u] == 1)[0]
                others = [v for v in neighbors_u if v != w]
                if others:
                    v = max(others, key=lambda x: degs[x])
                    adj[u, v] = 0
                    adj[v, u] = 0
                    degs[u] -= 1
                    degs[v] -= 1
            else:
                v = neighbors_u[np.argmax(degs[neighbors_u])]
                adj[u, v] = 0
                adj[v, u] = 0
                degs[u] -= 1
                degs[v] -= 1
                for x in range(n):
                    if x != w and not adj[w, x] and degs[x] < d:
                        adj[w, x] = 1
                        adj[x, w] = 1
                        degs[w] += 1
                        degs[x] += 1
                        break

    return adj


# ====================================================================
# Scramble methods
# ====================================================================

def init_icg_plus(n, d):
    """
    Inversive Congruential + quadratic perturbation.
    f(i,k) = (a * (i+1)^{-1} + b_k + c * i^2) mod p
    The i^2 term adds node-dependent nonlinearity on top of ICG's
    proven low serial correlation.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n + 1)
    a = primitive_root(p)
    c = int(p * phi) % p
    if c == 0:
        c = 1

    for k in range(d):
        b = int(p * ((k + 1) * phi)) % p
        for i in range(n):
            ip = (i + 1) % p
            if ip == 0:
                ip = 1
            inv = mod_inverse(ip, p)
            j = (a * inv + b + c * i * i) % p % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return degree_regularize(adj, d)


def init_power_perm(n, d):
    """
    Power permutation: f(i,k) = (i+1)^{e_k} mod p
    where e_k is coprime to p-1 → guaranteed bijection on Z_p*.
    Different exponents per layer.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n + 1)
    pm1 = p - 1

    # Find d exponents coprime to p-1, spread via golden ratio
    exponents = []
    for k in range(d):
        e = int(((k + 1) * phi * pm1) % pm1) + 2
        while np.gcd(e, pm1) != 1:
            e += 1
        exponents.append(e)

    for k in range(d):
        e = exponents[k]
        for i in range(n):
            j = pow(i + 1, e, p) % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return degree_regularize(adj, d)


def init_qr_scatter(n, d):
    """
    Quadratic residue scatter: f(i,k) = i*(i + c_k) mod p
    Not a bijection — produces ~2x edges (some collisions).
    Gives regularizer more material to work with.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n + 1)
    g = primitive_root(p)

    for k in range(d):
        c = pow(g, k + 1, p)
        for i in range(n):
            ip = (i + 1) % p
            if ip == 0:
                ip = 1
            j = (ip * (ip + c)) % p % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return degree_regularize(adj, d)


def init_composite(n, d):
    """
    Composition of two maps per layer:
    Step 1: x = (i+1)^{e_k} mod p      (power permutation — bijection)
    Step 2: j = (a * x^{-1} + b_k) mod p  (inversive — low correlation)
    Composition of two bijections = bijection. Double the mixing.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n + 1)
    pm1 = p - 1
    g = primitive_root(p)
    a = g

    for k in range(d):
        e = int(((k + 1) * phi * pm1) % pm1) + 2
        while np.gcd(e, pm1) != 1:
            e += 1
        b = pow(g, k + 1, p)
        for i in range(n):
            # Step 1: power permutation
            x = pow(i + 1, e, p)
            # Step 2: inversive
            if x == 0:
                x = 1
            inv_x = mod_inverse(x, p)
            j = (a * inv_x + b) % p % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return degree_regularize(adj, d)


def init_dlog_scatter(n, d):
    """
    Discrete-log inspired: f(i,k) = g^{(i+1)*(k+1)} mod p
    This maps sequential inputs to exponentially scattered outputs.
    Provably uniform on Z_p* (g is primitive root → full period).
    """
    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n + 1)
    g = primitive_root(p)

    for k in range(d):
        for i in range(n):
            exp = ((i + 1) * (k + 1)) % (p - 1)
            j = pow(g, exp, p) % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return degree_regularize(adj, d)


def init_random(n, d, seed=42):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for k in range(d):
        for i in range(n):
            j = rng.randint(0, n)
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return degree_regularize(adj, d)


# ====================================================================
# Run
# ====================================================================

configs = [
    # Even n (powers of 2)
    (32, 4), (64, 4), (128, 4), (256, 4), (512, 4),
    (32, 6), (64, 6), (128, 6), (256, 6), (512, 6),
    (64, 8), (128, 8), (256, 8), (512, 8),
    # Odd n (n*d must be even → d must be even)
    (25, 4), (33, 4), (49, 4), (63, 4), (65, 4), (99, 4), (127, 4), (255, 4), (511, 4),
    (25, 6), (33, 6), (49, 6), (63, 6), (99, 6), (127, 6), (255, 6),
    (49, 8), (63, 8), (99, 8), (127, 8), (255, 8),
    # Primes
    (31, 4), (37, 4), (61, 4), (67, 4), (97, 4), (127, 4), (251, 4), (509, 4),
    (31, 6), (37, 6), (61, 6), (67, 6), (97, 6),
    # Weird/awkward sizes
    (17, 4), (19, 4), (23, 4), (50, 4), (100, 4), (200, 4),
    (17, 6), (50, 6), (100, 6),
    # Odd d (n must be even)
    (32, 3), (64, 3), (128, 3), (256, 3),
    (32, 5), (64, 5), (128, 5), (256, 5),
    (64, 7), (128, 7), (256, 7),
]

methods = [
    ("QR-scat",   init_qr_scatter),
    ("Composite", init_composite),
    ("DLog",      init_dlog_scatter),
    ("Rand+REG",  init_random),
]

print(f"{'n':>5} {'d':>2} | {'RD':>7} {'Ram':>6}", end="")
for name, _ in methods:
    print(f" | {name:>12}", end="")
print()
print("-" * (22 + 16 * len(methods)))

for n, d in configs:
    if (n * d) % 2 != 0:
        continue

    ram = ramanujan(d)
    rd = rd_mean(n, d, 15)
    if rd == 0:
        rd = ram

    print(f"{n:5d} {d:2d} | {rd:7.4f} {ram:6.4f}", end="", flush=True)

    for name, builder in methods:
        adj = builder(n, d)
        degs = adj.sum(axis=1)
        if np.all(degs == d):
            l2 = lambda2(adj)
            pct = l2 / rd * 100
            mark = "+" if pct >= 100 else " "
            print(f" | {l2:6.3f}{pct:4.0f}%{mark}", end="", flush=True)
        else:
            mn, mx = int(degs.min()), int(degs.max())
            print(f" | FAIL[{mn},{mx}]  ", end="", flush=True)

    print()
