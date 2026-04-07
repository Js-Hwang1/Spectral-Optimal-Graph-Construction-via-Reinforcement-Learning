"""
Parallel degree regularization: match over-degree with under-degree
nodes in rounds, perform independent swaps simultaneously.

Each round:
  1. Find all over-degree and under-degree nodes
  2. Match them into independent pairs (u_over, w_under)
  3. For each pair: find v in N(u) not in N(w), transfer edge
  4. All transfers are independent → parallelizable

Test locally, then port to CUDA.
"""

import numpy as np
from scipy.linalg import eigvalsh
import time


def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


def random_init(n, d, seed=42):
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    while count < m:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1; adj[j, i] = 1; count += 1
    return adj


def sequential_regularize(adj, d):
    """Original sequential version for comparison."""
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)

    for _ in range(n * n):
        over = [i for i in range(n) if degs[i] > d]
        under = [i for i in range(n) if degs[i] < d]
        if not over and not under: break
        if not over or not under: break
        u = min(over, key=lambda i: (-degs[i], i))
        w = min(under, key=lambda i: (degs[i], i))
        nbs = sorted(v for v in range(n) if adj[u, v])
        cands = sorted(nbs, key=lambda v: (-degs[v], v))
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
            v = max(nbs, key=lambda x: (degs[x], -x))
            adj[u, v] = 0; adj[v, u] = 0; degs[u] -= 1; degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1; adj[x, w] = 1; degs[w] += 1; degs[x] += 1; break
    return adj


def parallel_regularize(adj, d):
    """
    Parallel degree regularization.

    Each round:
      1. Find over-degree (deg > d) and under-degree (deg < d) nodes
      2. Greedily match over↔under into independent pairs
      3. For each pair, find a transferable edge and swap
      All swaps in a round are independent → parallel

    Returns d-regular graph.
    """
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)

    max_rounds = n * d
    total_swaps = 0

    for round_num in range(max_rounds):
        over = np.where(degs > d)[0]
        under = np.where(degs < d)[0]

        if len(over) == 0 and len(under) == 0:
            break
        if len(over) == 0 or len(under) == 0:
            break

        # Sort: most over-degree first, most under-degree first
        over = over[np.argsort(-degs[over])]
        under = under[np.argsort(degs[under])]

        # Greedy matching: pair over[i] with under[i]
        num_pairs = min(len(over), len(under))

        # Track which nodes are involved in this round's swaps
        involved = set()
        pairs = []

        for idx in range(num_pairs):
            u = over[idx]
            w = under[idx]
            if u in involved or w in involved:
                continue
            pairs.append((u, w))
            involved.add(u)
            involved.add(w)

        # Execute all swaps in parallel (independent pairs)
        for u, w in pairs:
            # Find v in N(u) such that:
            # - v != w
            # - v not in involved (don't touch other swaps' nodes)
            # - {w, v} not in E
            best_v = -1
            best_vd = -1
            for k in range(n):
                if adj[u, k] == 0: continue
                v = k
                if v == w: continue
                if v in involved: continue  # safety: don't touch shared nodes
                if adj[w, v]: continue
                if degs[v] > best_vd:
                    best_vd = degs[v]
                    best_v = v

            if best_v >= 0:
                # Transfer: remove {u, best_v}, add {w, best_v}
                adj[u, best_v] = 0; adj[best_v, u] = 0
                adj[w, best_v] = 1; adj[best_v, w] = 1
                degs[u] -= 1
                degs[w] += 1
                total_swaps += 1
            else:
                # Fallback: try adding {u, w} directly if not adjacent
                if not adj[u, w] and w not in involved:
                    adj[u, w] = 1; adj[w, u] = 1
                    degs[u] += 1; degs[w] += 1
                    # Remove highest-deg neighbor of u (not w, not involved)
                    best_v = -1; best_vd = -1
                    for k in range(n):
                        if adj[u, k] == 0: continue
                        v = k
                        if v == w or v in involved: continue
                        if degs[v] > best_vd:
                            best_vd = degs[v]; best_v = v
                    if best_v >= 0:
                        adj[u, best_v] = 0; adj[best_v, u] = 0
                        degs[u] -= 1; degs[best_v] -= 1
                    total_swaps += 1

        if len(pairs) == 0:
            # No valid pairs found — fall back to sequential for stragglers
            # This handles edge cases where all candidates conflict
            for u in over:
                if degs[u] <= d: continue
                for w in under:
                    if degs[w] >= d: continue
                    for k in range(n):
                        if adj[u, k] == 0: continue
                        v = k
                        if v == w: continue
                        if adj[w, v]: continue
                        adj[u, v] = 0; adj[v, u] = 0
                        adj[w, v] = 1; adj[v, w] = 1
                        degs[u] -= 1; degs[w] += 1
                        total_swaps += 1
                        break
                    break

    return adj, round_num + 1, total_swaps


# ============================================================
# Test
# ============================================================

print("Parallel vs Sequential degree regularization")
print("=" * 85)
print(f"{'n':>5} {'d':>3} | {'seq_time':>8} {'par_time':>8} {'speedup':>7} | "
      f"{'par_rnds':>8} {'par_swaps':>9} | {'seq_λ₂':>7} {'par_λ₂':>7} {'Ram':>7} {'both_ok':>8}")
print("-" * 85)

for n in [64, 128, 256, 512, 1024]:
    for d in [4, 6]:
        if (n * d) % 2 != 0: continue
        adj = random_init(n, d, seed=42)

        # Sequential
        t0 = time.time()
        adj_seq = sequential_regularize(adj.copy(), d)
        t_seq = time.time() - t0
        l2_seq = lambda2(adj_seq)
        reg_seq = np.all(adj_seq.sum(axis=1) == d)

        # Parallel
        t0 = time.time()
        adj_par, rounds, swaps = parallel_regularize(adj.copy(), d)
        t_par = time.time() - t0
        l2_par = lambda2(adj_par)
        reg_par = np.all(adj_par.sum(axis=1) == d)

        ram = ramanujan(d)
        speedup = t_seq / t_par if t_par > 0 else 0
        both_ok = reg_seq and reg_par and l2_seq >= ram * 0.99 and l2_par >= ram * 0.99

        print(f"{n:5d} {d:3d} | {t_seq:8.3f} {t_par:8.3f} {speedup:7.1f}x | "
              f"{rounds:8d} {swaps:9d} | {l2_seq:7.4f} {l2_par:7.4f} {ram:7.4f} "
              f"{'YES' if both_ok else 'NO':>8}")
    print()
