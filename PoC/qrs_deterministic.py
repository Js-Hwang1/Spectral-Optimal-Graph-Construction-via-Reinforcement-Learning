"""
QRS-DR with fully deterministic regularization (canonical tie-breaking).

Every tie broken by lowest node index → any node running this algorithm
independently arrives at the exact same d-regular graph.

Compare against the original (numpy argmax tie-breaking) to measure
performance impact.
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


# ====================================================================
# QR Scatter (Phase 1) — identical for both versions
# ====================================================================

def qr_scatter(n, d):
    adj = np.zeros((n, n), dtype=np.int8)
    p = next_prime(n + 1)
    g = primitive_root(p)

    for k in range(d):
        c = pow(g, k + 1, p)
        for i in range(n):
            t = (i + 1) % p
            if t == 0:
                t = 1
            j = (t * (t + c)) % p % n
            if j == i:
                j = (j + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


# ====================================================================
# Original regularizer (numpy argmax — non-deterministic ties)
# ====================================================================

def regularize_original(adj, d):
    n = adj.shape[0]
    adj = adj.copy()
    np.fill_diagonal(adj, 0)
    adj = np.clip(adj, 0, 1)

    target_edges = n * d // 2
    degs = adj.sum(axis=1).astype(int)

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
# Deterministic regularizer (all ties broken by lowest index)
# ====================================================================

def regularize_deterministic(adj, d):
    """
    Fully deterministic degree regularization.
    All tie-breaking uses LOWEST NODE INDEX.
    Any node running this independently gets the exact same result.
    """
    n = adj.shape[0]
    adj = adj.copy()
    np.fill_diagonal(adj, 0)
    adj = np.clip(adj, 0, 1)

    target_edges = n * d // 2
    degs = adj.sum(axis=1).astype(int)

    # Step 1: Fix total edge count
    current_edges = int(adj.sum()) // 2

    while current_edges > target_edges:
        # Find highest-degree node (tie: lowest index)
        max_deg = int(degs.max())
        u = -1
        for i in range(n):
            if degs[i] == max_deg:
                u = i
                break

        # Among u's neighbors, find highest-degree (tie: lowest index)
        neighbors = sorted([v for v in range(n) if adj[u, v] == 1])
        if not neighbors:
            break
        best_deg = max(degs[v] for v in neighbors)
        v = -1
        for nb in neighbors:
            if degs[nb] == best_deg:
                v = nb
                break

        adj[u, v] = 0
        adj[v, u] = 0
        degs[u] -= 1
        degs[v] -= 1
        current_edges -= 1

    while current_edges < target_edges:
        # Find lowest-degree node (tie: lowest index)
        min_deg = int(degs.min())
        u = -1
        for i in range(n):
            if degs[i] == min_deg:
                u = i
                break

        # Find lowest-degree non-neighbor (tie: lowest index)
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

    # Step 2: Edge-swap to equalize degrees
    for _ in range(n * n):
        # Find most over-degree (tie: lowest index)
        over = [i for i in range(n) if degs[i] > d]
        under = [i for i in range(n) if degs[i] < d]
        if not over and not under:
            break

        if not over or not under:
            # Edge count mismatch — shouldn't happen, but handle gracefully
            break

        # Pick u = highest degree among over (tie: lowest index)
        max_over = max(degs[i] for i in over)
        u = -1
        for i in over:
            if degs[i] == max_over:
                u = i
                break

        # Pick w = lowest degree among under (tie: lowest index)
        min_under = min(degs[i] for i in under)
        w = -1
        for i in under:
            if degs[i] == min_under:
                w = i
                break

        # Direct transfer: find v in N(u), v != w, {w,v} not in E
        # Among candidates, prefer highest degree (tie: lowest index)
        neighbors_u = sorted([v for v in range(n) if adj[u, v] == 1])
        transferred = False

        # Sort candidates: highest degree first, then lowest index
        candidates = sorted(neighbors_u, key=lambda v: (-degs[v], v))
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

        if transferred:
            continue

        # Fallback: add {u,w} then remove worst edge from u
        if not adj[u, w]:
            adj[u, w] = 1
            adj[w, u] = 1
            degs[u] += 1
            degs[w] += 1
            # Remove edge from u to highest-degree neighbor != w (tie: lowest index)
            neighbors_u = sorted([v for v in range(n) if adj[u, v] == 1 and v != w])
            if neighbors_u:
                best_deg = max(degs[v] for v in neighbors_u)
                v = -1
                for nb in neighbors_u:
                    if degs[nb] == best_deg:
                        v = nb
                        break
                adj[u, v] = 0
                adj[v, u] = 0
                degs[u] -= 1
                degs[v] -= 1
        else:
            # u and w already adjacent — remove from u, add to w elsewhere
            # Remove: highest-degree neighbor of u (tie: lowest index)
            best_deg = max(degs[v] for v in neighbors_u)
            v = -1
            for nb in neighbors_u:
                if degs[nb] == best_deg:
                    v = nb
                    break
            adj[u, v] = 0
            adj[v, u] = 0
            degs[u] -= 1
            degs[v] -= 1
            # Add: connect w to lowest-degree non-neighbor (tie: lowest index)
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1
                    adj[x, w] = 1
                    degs[w] += 1
                    degs[x] += 1
                    break

    return adj


# ====================================================================
# Run comparison
# ====================================================================

configs = [
    (32, 4), (64, 4), (128, 4), (256, 4), (512, 4),
    (32, 6), (64, 6), (128, 6), (256, 6), (512, 6),
    (64, 8), (128, 8), (256, 8), (512, 8),
    # Odd n
    (25, 4), (33, 4), (49, 4), (63, 4), (99, 4), (127, 4), (255, 4),
    (25, 6), (49, 6), (63, 6), (99, 6), (127, 6),
    (49, 8), (63, 8), (99, 8), (127, 8),
    # Primes
    (31, 4), (37, 4), (61, 4), (67, 4), (97, 4), (251, 4), (509, 4),
    (31, 6), (37, 6), (61, 6), (97, 6),
    # Odd d
    (32, 3), (64, 3), (128, 3), (256, 3),
    (32, 5), (64, 5), (128, 5), (256, 5),
    (64, 7), (128, 7),
]

print(f"{'n':>5} {'d':>2} | {'RD':>7} {'Ram':>6} | {'Original':>12} | {'Determin':>12} | {'delta':>6}")
print("-" * 65)

wins_orig = 0
wins_det = 0
total = 0

for n, d in configs:
    if (n * d) % 2 != 0:
        continue

    ram = ramanujan(d)
    rd = rd_mean(n, d, 15)
    if rd == 0:
        rd = ram

    scatter = qr_scatter(n, d)

    # Original
    adj_o = regularize_original(scatter.copy(), d)
    degs_o = adj_o.sum(axis=1)
    reg_o = np.all(degs_o == d)
    l2_o = lambda2(adj_o) if reg_o else 0
    pct_o = l2_o / rd * 100 if rd > 0 else 0

    # Deterministic
    adj_d = regularize_deterministic(scatter.copy(), d)
    degs_d = adj_d.sum(axis=1)
    reg_d = np.all(degs_d == d)
    l2_d = lambda2(adj_d) if reg_d else 0
    pct_d = l2_d / rd * 100 if rd > 0 else 0

    delta = pct_d - pct_o

    if reg_o:
        o_str = f"{l2_o:6.3f}{pct_o:4.0f}%{'+'if pct_o>=100 else ' '}"
    else:
        o_str = f"FAIL[{int(degs_o.min())},{int(degs_o.max())}]"

    if reg_d:
        d_str = f"{l2_d:6.3f}{pct_d:4.0f}%{'+'if pct_d>=100 else ' '}"
    else:
        d_str = f"FAIL[{int(degs_d.min())},{int(degs_d.max())}]"

    print(f"{n:5d} {d:2d} | {rd:7.4f} {ram:6.4f} | {o_str:>12} | {d_str:>12} | {delta:+5.1f}%")

    if reg_o and pct_o >= 100:
        wins_orig += 1
    if reg_d and pct_d >= 100:
        wins_det += 1
    total += 1

print(f"\nBeats RD:  Original {wins_orig}/{total}  Deterministic {wins_det}/{total}")
