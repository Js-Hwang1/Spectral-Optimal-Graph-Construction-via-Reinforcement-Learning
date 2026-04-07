"""
Ablation: Does more degree imbalance in the init → better spectral gap?

Test inits with varying degree imbalance, all with exactly m=nd/2 edges.
Then apply the SAME regularizer to each. Measure λ₂ of the output.

Imbalance levels:
  1. Near-regular: each node gets ~d edges (low imbalance)
  2. Poisson-like: random edges (medium imbalance, our default)
  3. Power-law: few hubs with high degree, many leaves (high imbalance)
  4. Star-heavy: a few nodes get most edges (extreme imbalance)
  5. Bimodal: half nodes get 0, half get 2d (extreme)
  6. Single hub: one node gets all edges (maximum imbalance)
"""

import numpy as np
from scipy.linalg import eigvalsh


def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


def regularize(adj, d):
    """Same regularizer as always."""
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
# Init generators — all produce exactly m=nd/2 edges
# ============================================================

def init_uniform_random(n, d, seed=0):
    """Random edges, Poisson-like degree dist."""
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    while count < m:
        i = rng.randint(0, n); j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


def init_near_regular(n, d, seed=0):
    """Near-regular: round-robin edge assignment, very low imbalance."""
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    # Cycle through nodes, each picks one random neighbor
    attempts = 0
    while count < m:
        i = count % n
        candidates = [j for j in range(n) if j != i and adj[i, j] == 0]
        if candidates:
            j = rng.choice(candidates)
            adj[i, j] = 1; adj[j, i] = 1; count += 1
        attempts += 1
        if attempts > 10 * m: break
    return adj


def init_power_law(n, d, seed=0):
    """Power-law-ish: node i has probability ~ 1/(i+1) of being an endpoint."""
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    # Weight by 1/(i+1)
    weights = 1.0 / (np.arange(n) + 1.0)
    weights /= weights.sum()
    count = 0
    while count < m:
        i = rng.choice(n, p=weights)
        j = rng.choice(n, p=weights)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


def init_hubs(n, d, seed=0, num_hubs=None):
    """Hub-heavy: a few hub nodes get most edges."""
    m = n * d // 2
    if num_hubs is None: num_hubs = max(2, n // 8)
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    while count < m:
        # 80% of edges involve a hub
        if rng.random() < 0.8:
            i = rng.randint(0, num_hubs)
        else:
            i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


def init_bimodal(n, d, seed=0):
    """Bimodal: first n/4 nodes get ~3d edges each, rest get ~d/3 each."""
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    heavy = n // 4
    count = 0
    while count < m:
        if rng.random() < 0.75:
            i = rng.randint(0, heavy)  # heavy nodes
        else:
            i = rng.randint(heavy, n)  # light nodes
        j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


def init_chain_star(n, d, seed=0):
    """Chain + star: first half are star centers, second half are leaves."""
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    stars = n // 4
    count = 0
    while count < m:
        # Star: connect star center to random leaf
        if rng.random() < 0.6:
            i = rng.randint(0, stars)
            j = rng.randint(stars, n)
        else:
            i = rng.randint(0, n)
            j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


def init_one_clique(n, d, seed=0):
    """One dense clique + sparse rest: first sqrt(n) nodes form dense subgraph."""
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    clique = max(4, int(np.sqrt(n)))
    count = 0
    while count < m:
        if rng.random() < 0.7:
            i = rng.randint(0, clique)
            j = rng.randint(0, clique)
        else:
            i = rng.randint(0, n)
            j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


# ============================================================
# Run ablation
# ============================================================

inits = [
    ("Near-regular", init_near_regular),
    ("Uniform random", init_uniform_random),
    ("Power-law", init_power_law),
    ("Hub-heavy", init_hubs),
    ("Bimodal", init_bimodal),
    ("Star-leaf", init_chain_star),
    ("One-clique", init_one_clique),
]

for n in [64, 128, 256]:
    d = 4
    if (n * d) % 2 != 0: continue
    m = n * d // 2
    ram = ramanujan(d)

    print(f"\nn={n}, d={d}, m={m}, Ram={ram:.4f}")
    print(f"{'Init':>15} | {'deg_min':>7} {'deg_max':>7} {'deg_std':>7} {'deg_cv':>7} | "
          f"{'λ₂(init)':>9} {'λ₂(reg)':>8} {'ratio':>6}")
    print("-" * 80)

    for name, fn in inits:
        # Average over 3 seeds
        l2_regs = []
        for seed in range(3):
            adj = fn(n, d, seed=seed)
            edges = int(adj.sum()) // 2
            if edges != m:
                continue
            degs = adj.sum(axis=1).astype(float)
            deg_std = degs.std()
            deg_cv = deg_std / degs.mean() if degs.mean() > 0 else 0

            l2_init = lambda2(adj)
            adj_r = regularize(adj, d)
            is_reg = np.all(adj_r.sum(axis=1) == d)
            if is_reg:
                l2_regs.append(lambda2(adj_r))

        if l2_regs:
            # Use last seed's stats for display
            print(f"{name:>15} | {int(degs.min()):7d} {int(degs.max()):7d} "
                  f"{deg_std:7.2f} {deg_cv:7.2f} | "
                  f"{l2_init:9.4f} {np.mean(l2_regs):8.4f} {np.mean(l2_regs)/ram:6.2f}")
        else:
            print(f"{name:>15} | FAILED")

print("\n\ndeg_cv = coefficient of variation (std/mean) — higher = more imbalanced")
