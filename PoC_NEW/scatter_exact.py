"""
Goal: deterministic scatter that produces EXACTLY m = nd/2 undirected edges.
Each node computes independently. No communication.

Approach: each node i picks d/2 targets j_0, j_1, ..., j_{d/2-1} via
the QR formula. Each (i, j_k) is a directed edge. After ALL nodes
compute their targets, we symmetrize: edge {i,j} exists iff i targeted
j OR j targeted i (or both).

With n nodes each producing d/2 directed edges = nd/2 directed edges total.
After symmetrization, if there are no "mutual" edges (i targets j AND j
targets i), we get exactly nd/2 undirected edges.
If there ARE mutual edges, we get fewer (each mutual pair counts once).

So: nd/2 directed → between nd/4 and nd/2 undirected.
We want nd/2 undirected.

Alternative: each node picks d targets (not d/2). nd directed edges.
After symmetrization with NO mutuals: nd undirected. Too many.
With ALL mutual: nd/2 undirected. Perfect!

So the question is: can we design the formula so that EVERY directed
edge has a mutual counterpart? i.e., if i targets j, then j targets i
(possibly in a different layer)?

This is equivalent to: the scatter map is an INVOLUTION (self-inverse
permutation) within each layer, OR edges pair up across layers.

Let's test empirically: how many mutuals does the QR formula produce?
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


# ============================================================
# Approach 1: d/2 layers, each node picks 1 target per layer
# ============================================================

def scatter_d2(n, d):
    """d/2 layers, symmetrize."""
    p = next_prime(n)
    g = primitive_root(p)
    num_layers = d // 2

    # Collect directed edges per layer
    directed = set()
    for k in range(num_layers):
        c = pow(g, k + 1, p)
        for i in range(n):
            t = (i + 1) % p
            if t == 0: t = 1
            j = (t * (t + c)) % p % n
            if j == i: j = (j + 1) % n
            directed.add((i, j))

    # Count mutuals
    mutuals = sum(1 for (a, b) in directed if (b, a) in directed) // 2
    undirected = set()
    for (a, b) in directed:
        undirected.add((min(a, b), max(a, b)))

    return directed, undirected, mutuals


# ============================================================
# Approach 2: d layers, ONLY keep edge if BOTH directions exist
# (mutual-only symmetrization)
# ============================================================

def scatter_mutual_only(n, d):
    """d layers, only keep mutual edges."""
    p = next_prime(n)
    g = primitive_root(p)

    directed = set()
    for k in range(d):
        c = pow(g, k + 1, p)
        for i in range(n):
            t = (i + 1) % p
            if t == 0: t = 1
            j = (t * (t + c)) % p % n
            if j == i: j = (j + 1) % n
            directed.add((i, j))

    # Keep only mutual edges
    undirected = set()
    for (a, b) in directed:
        if (b, a) in directed:
            undirected.add((min(a, b), max(a, b)))

    return directed, undirected


# ============================================================
# Approach 3: d/2 layers, each node picks 1 target.
# Then ALSO add the reverse: for each (i,j), add (j,i).
# This guarantees every directed edge has a mutual.
# But then we have n*d/2 mutual pairs = n*d/2 undirected edges? No.
# We have n*(d/2) directed edges. Adding reverses gives 2*n*(d/2) = nd
# directed edges, but they form nd/2 mutual pairs = nd/2 undirected. YES!
# BUT: some reverses may collide with existing edges.
# ============================================================

# Actually the simplest correct approach:
# Each node picks d/2 targets. That's nd/2 directed edges.
# Symmetrize: each directed edge becomes undirected.
# If i→j and j→i both exist (mutual), they become 1 undirected edge.
# If i→j but NOT j→i, it's still 1 undirected edge.
# So # undirected = # unique {i,j} pairs = # directed - # mutuals.
# We want # undirected = nd/2.
# # directed = nd/2.
# So # undirected = nd/2 - mutuals.
# If mutuals = 0: perfect, nd/2 undirected.
# If mutuals > 0: we have FEWER than nd/2. Deficit.

# ============================================================
# Test all approaches
# ============================================================

print(f"{'n':>5} {'d':>3} {'target':>7} | "
      f"{'d/2: dir':>8} {'undir':>6} {'mutual':>7} {'Δ':>4} | "
      f"{'mutual_only: undir':>18} {'Δ':>4}")
print("-" * 75)

for n in [32, 64, 128, 256]:
    for d in [4, 6, 8]:
        if (n * d) % 2 != 0: continue
        target = n * d // 2

        # Approach 1
        dir1, undir1, mut1 = scatter_d2(n, d)
        delta1 = len(undir1) - target

        # Approach 2
        dir2, undir2 = scatter_mutual_only(n, d)
        delta2 = len(undir2) - target

        print(f"{n:5d} {d:3d} {target:7d} | "
              f"{len(dir1):8d} {len(undir1):6d} {mut1:7d} {delta1:4d} | "
              f"{len(undir2):18d} {delta2:4d}")
    print()
