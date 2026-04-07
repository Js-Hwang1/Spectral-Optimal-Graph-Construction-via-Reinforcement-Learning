"""
Involution-based scatter: EXACTLY nd/2 edges, ZERO collisions, guaranteed.

An involution π on [0, n-1] is a permutation where π(π(i)) = i for all i.
It consists of n/2 2-cycles (swaps) and no fixed points (derangement).
Each 2-cycle (i, π(i)) is one undirected edge. n/2 edges per involution.

d/2 edge-disjoint involutions → d/2 × n/2 = nd/4 edges. NOT ENOUGH.

d edge-disjoint involutions → d × n/2 = nd/2 edges. EXACT.

So: d involutions, each a fixed-point-free involution on [0, n-1],
pairwise edge-disjoint. Each involution = perfect matching = n/2 edges.
d matchings × n/2 = nd/2 edges. ZERO collisions by construction.

This is EXACTLY a 1-factorization of a d-regular graph!

But we need to BUILD these involutions analytically.
The round-robin tournament gives n-1 involutions on n nodes (n even).
Pick d of them → d-regular graph with exactly nd/2 edges.

Each involution is defined analytically:
  Round-robin round r (r = 0, ..., n-2):
    Node 0 pairs with node r+1 (mod n-1) mapped to [1, n-1]
    Node at position k pairs with node at position (n-1) - k
    (positions rotate by r)

This is fully analytical, per-node computable, zero collisions.
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


def round_robin_matching(n, r):
    """
    Round-robin tournament: perfect matching for round r.
    n must be even. Returns partner array: partner[i] = j means i↔j.
    Produces exactly n/2 pairs.

    Algorithm (standard):
      - Node n-1 is fixed ("phantom"). Nodes 0..n-2 rotate.
      - In round r: node n-1 pairs with node (r mod (n-1)).
      - For k = 1, ..., n/2-1: node (r+k) mod (n-1) pairs with
        node (r-k) mod (n-1).
    """
    assert n % 2 == 0
    partner = np.full(n, -1, dtype=int)
    m = n - 1  # number of rotating nodes

    # Fixed node n-1 pairs with rotating node r mod m
    a = r % m
    partner[n - 1] = a
    partner[a] = n - 1

    # Remaining pairs
    for k in range(1, n // 2):
        u = (r + k) % m
        v = (r - k) % m
        partner[u] = v
        partner[v] = u

    return partner


def scatter_involution(n, d, selection="spread"):
    """
    Build d involutions (perfect matchings) from round-robin tournament.
    Selection strategy determines WHICH d of the n-1 available matchings.

    Returns adjacency matrix with exactly nd/2 edges.
    """
    assert n % 2 == 0 and d >= 2 and d < n

    total_rounds = n - 1  # n-1 available matchings

    if selection == "spread":
        # Evenly spread: pick matchings at indices 0, (n-1)/d, 2(n-1)/d, ...
        indices = [int(round(k * (total_rounds - 1) / (d - 1))) if d > 1 else 0
                   for k in range(d)]
        # Deduplicate
        indices = sorted(set(indices))
        while len(indices) < d:
            for r in range(total_rounds):
                if r not in indices:
                    indices.append(r)
                    break
            indices = sorted(set(indices))
        indices = indices[:d]
    elif selection == "golden":
        # Golden ratio spacing
        phi = (1 + 5**0.5) / 2
        indices = sorted(set(int(k * phi * total_rounds) % total_rounds
                             for k in range(d)))
        while len(indices) < d:
            for r in range(total_rounds):
                if r not in indices:
                    indices.append(r)
                    break
            indices = sorted(indices)
        indices = indices[:d]
    else:
        indices = list(range(d))

    # Build adjacency from selected matchings
    adj = np.zeros((n, n), dtype=np.int8)
    edge_count = 0

    for r in indices:
        partner = round_robin_matching(n, r)
        for i in range(n):
            j = partner[i]
            if j > i:  # count each edge once
                assert adj[i, j] == 0, f"Collision at round {r}: ({i},{j}) already exists"
                adj[i, j] = 1
                adj[j, i] = 1
                edge_count += 1

    return adj, edge_count, indices


def regularize(adj, d):
    """Degree equalization via edge swaps (preserves edge count)."""
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)
    current = int(adj.sum()) // 2
    target = n * d // 2
    assert current == target, f"edges={current}, need={target}"

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
            adj[u, w] = 1; adj[w, u] = 1; degs[u] += 1; degs[w] += 1
            others = sorted(v for v in range(n) if adj[u, v] and v != w)
            if others:
                bd = max(degs[v] for v in others)
                v = next(nb for nb in others if degs[nb] == bd)
                adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
        else:
            v = max(neighbors_u, key=lambda x: (degs[x], -x))
            adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1; adj[x, w] = 1; degs[w] += 1; degs[x] += 1; break
    return adj


# ============================================================
# Test
# ============================================================

print("Involution-based scatter: d matchings from round-robin tournament")
print("GUARANTEES: exactly nd/2 edges, zero collisions, d-regular after regularization")
print("=" * 85)

for selection in ["spread", "golden", "sequential"]:
    print(f"\n--- Selection: {selection} ---")
    print(f"{'n':>5} {'d':>3} {'target':>7} | {'edges':>6} {'Δ':>3} | "
          f"{'λ₂(scat)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
    print("-" * 70)

    for n in [32, 64, 128, 256]:
        for d in [4, 6, 8]:
            if (n * d) % 2 != 0 or d >= n: continue
            target = n * d // 2

            try:
                adj_s, edges, indices = scatter_involution(n, d, selection)
            except Exception as e:
                print(f"{n:5d} {d:3d} {target:7d} | ERROR: {e}")
                continue

            delta = edges - target

            # The involution scatter is ALREADY d-regular (d matchings = d-regular)
            # No regularization needed!
            degs_s = adj_s.sum(axis=1)
            is_reg_before = np.all(degs_s == d)

            l2_s = lambda2(adj_s)
            ram = ramanujan(d)

            if is_reg_before:
                print(f"{n:5d} {d:3d} {target:7d} | {edges:6d} {delta:3d} | "
                      f"{l2_s:9.4f} {'=scat':>8s} {ram:8.4f} {l2_s/ram:6.2f} "
                      f"{'YES':>5s}  (already d-regular!)")
            else:
                adj_r = regularize(adj_s, d)
                is_reg = np.all(adj_r.sum(axis=1) == d)
                l2_r = lambda2(adj_r)
                print(f"{n:5d} {d:3d} {target:7d} | {edges:6d} {delta:3d} | "
                      f"{l2_s:9.4f} {l2_r:8.4f} {ram:8.4f} {l2_r/ram:6.2f} "
                      f"{'YES' if is_reg else 'NO':>5s}")
