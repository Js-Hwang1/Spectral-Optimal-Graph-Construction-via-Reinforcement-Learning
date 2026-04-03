#!/usr/bin/env python3
"""
PoC: Ring init, pure greedy ADD (no swaps). FV vs SO vs DB baselines.
Matches exactly how DB baselines work: start from ring, greedily add m-n edges.
"""
import numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.gnm_env import algebraic_connectivity

def build_ring(n):
    adj = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1; adj[j, i] = 1
    return adj

def full_eigen(adj):
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    return np.linalg.eigh(L)

def greedy_add_fv(n, m):
    """FV greedy: add edge with max (v2[i]-v2[j])^2, starting from ring."""
    adj = build_ring(n)
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    for _ in range(m - n):
        vals, vecs = full_eigen(adj)
        v2 = vecs[:, 1]
        fg = (v2[:, None] - v2[None, :]) ** 2
        mask = (adj == 0) & tri
        if not mask.any():
            break
        scores = np.where(mask, fg, -np.inf)
        best = int(np.argmax(scores.ravel()))
        i, j = best // n, best % n
        adj[i, j] = 1; adj[j, i] = 1
    return algebraic_connectivity(adj)

def greedy_add_so(n, m):
    """SO greedy: second-order perturbation scoring, starting from ring."""
    adj = build_ring(n)
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    for _ in range(m - n):
        vals, vecs = full_eigen(adj)
        v2 = vecs[:, 1]
        lam2 = vals[1]
        fg = (v2[:, None] - v2[None, :]) ** 2
        n_eig = min(n - 1, 8)
        corr = np.zeros((n, n))
        for k in range(2, n_eig + 1):
            vk = vecs[:, k]
            gap = max(vals[k] - lam2, 1e-10)
            corr += (vk[:, None] - vk[None, :]) ** 2 / gap
        score = fg * (1.0 - corr)
        mask = (adj == 0) & tri
        if not mask.any():
            break
        scores = np.where(mask, score, -np.inf)
        best = int(np.argmax(scores.ravel()))
        i, j = best // n, best % n
        adj[i, j] = 1; adj[j, i] = 1
    return algebraic_connectivity(adj)

cache = json.load(open(Path(__file__).parent.parent / 'baselines_cache.json'))

for n in [8, 10, 16, 24]:
    nk = str(n)
    if nk not in cache:
        continue
    nd = cache[nk]
    max_m = n * (n - 1) // 2
    all_m = list(range(n, max_m + 1))
    if len(all_m) > 20:
        step = max(1, len(all_m) // 20)
        ms = all_m[::step]
        if all_m[-1] not in ms:
            ms.append(all_m[-1])
    else:
        ms = all_m

    print(f"\nn={n}")
    print(f"{'(n,m)':>9}  {'DB-FV':>9} {'DB-ER':>9} {'DB-best':>9} {'FV-add':>9} {'SO-add':>9}  {'diff':>8}  note")
    print("-" * 100)
    so_w, fv_w, so_db, fv_db = 0, 0, 0, 0
    tot = 0
    for m in ms:
        bl = nd.get(str(m), {})
        if not bl:
            continue
        db_fv = bl.get('fv', 0) or 0
        db_er = bl.get('er', 0) or 0
        db_best = max(db_fv, db_er,
                      bl.get('sw_025', 0) or 0,
                      bl.get('sw_050', 0) or 0,
                      bl.get('sw_075', 0) or 0)

        fv = greedy_add_fv(n, m)
        so = greedy_add_so(n, m)
        d = so - fv
        tot += 1

        tags = []
        if so > fv + 1e-6:
            so_w += 1
            tags.append("SO>FV")
        elif fv > so + 1e-6:
            fv_w += 1
            tags.append("FV>SO")
        if so > db_best + 1e-6:
            so_db += 1
            tags.append("SO>DB!")
        if fv > db_best + 1e-6:
            fv_db += 1
            tags.append("FV>DB!")
        if abs(fv - db_fv) < 1e-4:
            tags.append("FV=DB")

        tag = " ".join(tags)
        print(f"({n},{m:>3})  {db_fv:9.4f} {db_er:9.4f} {db_best:9.4f} {fv:9.4f} {so:9.4f}  {d:+8.4f}  {tag}")

    print(f"\n  n={n}: SO>FV={so_w}/{tot}  FV>SO={fv_w}/{tot}  SO>DB={so_db}/{tot}  FV=DB(our)>DB(stored)={fv_db}/{tot}")
