"""
Random init (n, m=nd/2) + degree regularization.
Sanity check: Phase 1 produces EXACTLY m edges, Phase 2 makes d-regular.
"""
import numpy as np
from scipy.linalg import eigvalsh


def random_init(n, d, seed=42):
    """
    Phase 1: Random initialization with EXACTLY m = nd/2 edges.
    Each edge is chosen uniformly at random (no self-loops, no duplicates).
    """
    m = n * d // 2
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    count = 0
    while count < m:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1
            adj[j, i] = 1
            count += 1
    return adj


def regularize(adj, d):
    """
    Phase 2: Deterministic degree regularization.
    Precondition: adj has exactly nd/2 edges.
    Only does degree equalization (edge swaps), NO edge count adjustment.
    """
    n = adj.shape[0]
    adj = adj.copy()
    degs = adj.sum(axis=1).astype(int)

    # Verify precondition
    current_edges = int(adj.sum()) // 2
    target_edges = n * d // 2
    assert current_edges == target_edges, \
        f"Edge count mismatch: have {current_edges}, need {target_edges}"

    # Equalize degrees via edge swaps (preserves total edge count)
    for _ in range(n * n):
        over = [i for i in range(n) if degs[i] > d]
        under = [i for i in range(n) if degs[i] < d]
        if not over and not under:
            break
        if not over or not under:
            # Shouldn't happen if edge count is correct
            break

        u = min(over, key=lambda i: (-degs[i], i))
        w = min(under, key=lambda i: (degs[i], i))

        # Direct transfer: find v in N(u), v != w, {w,v} not in E
        neighbors_u = sorted(v for v in range(n) if adj[u, v])
        candidates = sorted(neighbors_u, key=lambda v: (-degs[v], v))
        transferred = False
        for v in candidates:
            if v == w:
                continue
            if not adj[w, v]:
                # Transfer: remove {u,v}, add {w,v}
                adj[u, v] = 0; adj[v, u] = 0
                adj[w, v] = 1; adj[v, w] = 1
                degs[u] -= 1; degs[w] += 1
                transferred = True
                break
        if transferred:
            continue

        # Fallback: add {u,w} then remove worst edge from u
        if not adj[u, w]:
            adj[u, w] = 1; adj[w, u] = 1
            degs[u] += 1; degs[w] += 1
            others = sorted(v for v in range(n) if adj[u, v] and v != w)
            if others:
                best_deg = max(degs[v] for v in others)
                v = next(nb for nb in others if degs[nb] == best_deg)
                adj[u, v] = 0; adj[v, u] = 0
                degs[u] -= 1; degs[v] -= 1
        else:
            # u and w already adjacent
            v = max(neighbors_u, key=lambda x: (degs[x], -x))
            adj[u, v] = 0; adj[v, u] = 0
            degs[u] -= 1; degs[v] -= 1
            for x in range(n):
                if x != w and not adj[w, x] and degs[x] < d:
                    adj[w, x] = 1; adj[x, w] = 1
                    degs[w] += 1; degs[x] += 1
                    break

    return adj


def lambda2(adj):
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1).astype(float))
    L = D - adj.astype(float)
    return eigvalsh(L)[1]


def ramanujan(d):
    return d - 2 * np.sqrt(d - 1)


if __name__ == "__main__":
    print("Random init (n, m=nd/2) + degree regularization")
    print("=" * 75)
    print(f"{'n':>5} {'d':>3} {'m':>6} | {'edges_init':>10} {'edges_reg':>10} | "
          f"{'λ₂(init)':>9} {'λ₂(reg)':>8} {'Ram':>8} {'ratio':>6} {'reg?':>5}")
    print("-" * 75)

    for n in [32, 64, 128, 256]:
        for d in [4, 6, 8]:
            if (n * d) % 2 != 0:
                continue
            m = n * d // 2

            adj_init = random_init(n, d, seed=42)
            edges_init = int(adj_init.sum()) // 2

            adj_reg = regularize(adj_init, d)
            edges_reg = int(adj_reg.sum()) // 2
            degs = adj_reg.sum(axis=1)
            is_reg = np.all(degs == d)

            l2_init = lambda2(adj_init)
            l2_reg = lambda2(adj_reg)
            ram = ramanujan(d)

            print(f"{n:5d} {d:3d} {m:6d} | {edges_init:10d} {edges_reg:10d} | "
                  f"{l2_init:9.4f} {l2_reg:8.4f} {ram:8.4f} {l2_reg/ram:6.2f} "
                  f"{'YES' if is_reg else 'NO':>5s}")
        print()
