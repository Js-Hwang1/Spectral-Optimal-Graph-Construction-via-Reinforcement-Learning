"""
python3 src/train_phase1.py --infer --load_model models_phase1/p1_spectral_psi_comp_ep00400.pt --n 48 --m 264 --device cpu
"""

import os
import concurrent.futures as _fut
import re
import math
import argparse
import random
from typing import List, Tuple, Dict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh as scipy_eigsh
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


OPTIMIZE = False
STRICT_MASK = True  # enable provable (constructive) feasibility mask
## (topk prefilter knobs removed)

# Structural biases to push toward high-λ2 within regular graphs (no lookahead)
USE_STRUCTURAL_BIAS = True
DENSE_PART_THRESH = 0.55   # when k/(n-1) >= this, favor multipartite cross edges
SPARSE_TRI_THRESH = 0.45   # when k/(n-1) <= this, discourage triangle creation
MULTI_MAX_PARTS = 4        # cap target multipartite parts for clustering proxy (2..4)
W_ER = 1.0                 # weight on effective-resistance score
W_PHI = 0.7                # weight on Fiedler gap (when connected)
W_PART = 0.6               # weight on cross-cluster edges in dense regime
W_TRI_SPARSE = 0.3         # penalty on common-neighbors (triangles) in sparse regime
W_C4_SPARSE = 0.25         # penalty on 4-cycles (approx) in sparse/mid regime
C4_THRESH = 0.55           # apply 4-cycle penalty when k/(n-1) <= this

# Remove ER dependency during training/inference (features + scoring)
USE_ER = False

# Spectral shaping (no lookahead): reinforce stepwise λ2 gains and Ramanujan margin
STEP_LAM2_WEIGHT = 0.5
RAMANUJAN_MARGIN_WEIGHT = 0.2

# Learned Fiedler surrogate (ψ) — auxiliary loss and feature
PSI_AUX_WEIGHT = 0.1
PSI_ORTH_W = 1.0
PSI_NORM_W = 0.1

# Complement-aware guidance for mid/high density (operate via features/shaping; no inference tricks)
COMPLEMENT_BIAS_THRESH = 0.58
W_PHI_COMP = 0.5
COMP_STEP_LAM2_WEIGHT = 0.3
COMP_RAMANUJAN_MARGIN_WEIGHT = 0.1

# Additional architectural and sampling options
USE_PAIRWISE_REL = True      # add |xu-xv| and xu*xv to scorer inputs
NODE_USE_LAPPE = True        # append Laplacian PE (phi2, phi3) to node features
# Density bucketed sampling (5 buckets by default); can use MPI to shard buckets
BUCKETS = 5
USE_MPI = False
TOPK_FRAC_DEFAULT = 0.33     # prune to this fraction of feasible candidates using φ-gap
MAX_MASK_SHORTLIST = 64      # cap number of candidates to run feasibility on per step

# Speed knobs for spectral terms
SPECTRAL_STEP_EVERY = 4      # compute step Δλ2 every N steps (or near end); else skip

# Feasibility checker configuration
FEAS_METHOD = 'auto'         # 'auto' tries bipartite f-factor first, falls back to DFS; 'dfs' forces backtracking
FEAS_PARALLEL = False        # set True to evaluate shortlist candidates in parallel (per process)
FEAS_WORKERS = max(1, min(4, os.cpu_count() or 1))

# ψ-weighted k-factor teacher (approximate) to push λ2 via global pairing
KFACTOR_TEACH = True
KFACTOR_TEACH_WEIGHT = 0.3
KFACTOR_TAU = 1.0

# Enumeration teacher (exact supervision on small n if available)
ENUM_TEACH = True
ENUM_TEACH_WEIGHT = 0.5

# Extra spectral gain in sparse regime (dens ≤ 0.4)
SPARSE_STEP_LAM2_GAIN = 0.5
SPARSE_RAMANUJAN_GAIN = 0.2

# Training guidance knobs (distil optimization-time moves into the policy)
TEACHER_FORCE = True
TEACHER_EDGES_LEFT_FRAC = 0.3
TEACHER_MIN_EDGES_LEFT = 8
TEACHER_KL_WEIGHT = 0.1
PHI_SHAPING_WEIGHT = 0.05
PHI_SHAPING_FRAC = 0.3
SWITCH_REWARD_WEIGHT = 5.0

# Structural distillation to train the policy (no lookahead)
STRUCT_TEACH = True
STRUCT_TEACH_WEIGHT = 0.2
STRUCT_TEACH_TAU = 1.0

# -----------------------------
# Graph utilities (standalone)
# -----------------------------

def laplacian_from_adj(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1)
    return np.diag(deg) - adj


def is_connected_adj(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    if n == 0:
        return True
    if int(adj.sum() // 2) < n - 1:
        return False
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    A = adj.astype(bool)
    while stack:
        u = stack.pop()
        nbrs = np.nonzero(A[u])[0]
        for v in nbrs:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    return bool(seen.all())


def laplacian_pseudoinverse(L: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(L)
    tol = 1e-12
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    G = (V * invw) @ V.T
    G = 0.5 * (G + G.T)
    return G


def G_from_adj(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    G = laplacian_pseudoinverse(laplacian_from_adj(adj))
    return G, np.diag(G).copy()


def lam2_smallest(L: np.ndarray, need_vec: bool = False):
    """Return λ2 (and optionally φ2) using a fast method when SciPy is available."""
    n = L.shape[0]
    if _HAVE_SCIPY and n > 4:
        try:
            w, v = scipy_eigsh(csr_matrix(L), k=min(3, n-1), which='SM', tol=1e-6, maxiter=max(1000, 10*n))
            idx = np.argsort(w)
            w = w[idx]; v = v[:, idx]
            lam2 = float(w[1]) if len(w) > 1 else 0.0
            if need_vec:
                phi2 = v[:, 1] if v.shape[1] > 1 else np.zeros(n)
                return lam2, phi2
            return lam2
        except Exception:
            pass
    # fallback
    w, v = np.linalg.eigh(L)
    idx = np.argsort(w)
    w = w[idx]; v = v[:, idx]
    lam2 = float(w[1]) if len(w) > 1 else 0.0
    if need_vec:
        phi2 = v[:, 1] if v.shape[1] > 1 else np.zeros(n)
        return lam2, phi2
    return lam2

def laplacian_small_eigvecs(L: np.ndarray, num: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return (w, V) for the smallest `num` Laplacian eigenpairs (ascending),
    including the trivial zero eigenpair when present. Falls back to dense eigh.
    """
    n = L.shape[0]
    num = max(1, min(num, n))
    if _HAVE_SCIPY and n > 4:
        try:
            w, v = scipy_eigsh(csr_matrix(L), k=min(num, n-1), which='SM', tol=1e-6, maxiter=max(1000, 10*n))
            idx = np.argsort(w)
            return w[idx], v[:, idx]
        except Exception:
            pass
    w, v = np.linalg.eigh(L)
    idx = np.argsort(w)
    return w[idx], v[:, idx]


def complement_adj(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    C = np.ones_like(A) - np.eye(n, dtype=A.dtype) - A
    C[C < 0] = 0.0
    return C


def _approx_kfactor_from_psi(A: np.ndarray, k: int, psi: np.ndarray) -> set[tuple[int, int]]:
    """Greedy ψ-gap weighted k-factor construction on the residual graph.
    Returns a set of edges to add (as (u,v) with u<v)). Uses EG feasibility to avoid dead-ends.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    r = (k - deg).astype(np.int64)
    E = set()
    if np.all(r == 0):
        return E
    # candidate non-edges with residual capacity
    uu, vv = np.triu_indices(n, k=1)
    mask = (A[uu, vv] == 0) & (r[uu] > 0) & (r[vv] > 0)
    uu = uu[mask]; vv = vv[mask]
    if uu.size == 0:
        return E
    w = (psi[uu] - psi[vv]) ** 2
    order = np.argsort(-w)
    uu = uu[order]; vv = vv[order]
    A2 = A.copy()
    for u, v in zip(uu, vv):
        if r[u] <= 0 or r[v] <= 0 or A2[u, v] == 1.0:
            continue
        r2 = r.copy(); r2[u] -= 1; r2[v] -= 1
        if np.any(r2 < 0):
            continue
        if not _is_graphical_erdos_gallai(r2):
            continue
        # accept
        A2[u, v] = A2[v, u] = 1.0
        r = r2
        E.add((int(min(u, v)), int(max(u, v))))
        if np.all(r == 0):
            break
    return E


def _is_graphical_erdos_gallai(seq_in: np.ndarray) -> bool:
    s = np.asarray(seq_in, dtype=np.int64)
    s = s[s > 0]
    if s.size == 0:
        return True
    if np.any(s < 0):
        return False
    n = int(s.size)
    if np.max(s) > n - 1:
        return False
    if (np.sum(s) & 1) != 0:
        return False
    s.sort(); s = s[::-1]
    prefix = np.cumsum(s)
    for k in range(1, n + 1):
        left = int(prefix[k - 1])
        if k < n:
            tail = s[k:]
            rhs = k * (k - 1) + int(np.minimum(tail, k).sum())
        else:
            rhs = k * (k - 1)
        if left > rhs:
            return False
    return True


def G_rank1_update_edge(G: np.ndarray, diagG: np.ndarray, u: int, v: int, w: float = 1.0):
    g = G[:, u] - G[:, v]
    R_uv = float(G[u, u] + G[v, v] - 2.0 * G[u, v])
    denom = (1.0 / w) + R_uv
    if denom <= 0:
        return G, diagG
    outer = np.outer(g, g) / denom
    G2 = G - outer
    G2 = 0.5 * (G2 + G2.T)
    diagG2 = diagG - (g * g) / denom
    return G2, diagG2


## (removed er_topk_from_G; no top‑k pruning)


def normalized_density(n: int, m: int) -> float:
    Mmax = n * (n - 1) / 2.0
    base = float(n - 1)
    den = max(1.0, Mmax - base)
    return float(min(1.0, max(0.0, (float(m) - base) / den)))


# -----------------------------
# Phase‑1 environment: build k‑regular graph from empty
# -----------------------------

class Phase1Env:
    def __init__(self, n: int, k: int):
        assert 0 <= k < n, 'k must be in [0,n)'
        self.n = n
        self.k = k
        self.m = (n * k) // 2
        self.reset()

    def reset(self):
        n = self.n
        self.adj = np.zeros((n, n), dtype=np.float64)
        self.t = 0
        return self.state()

    def state(self):
        n, k = self.n, self.k
        e_now = int(self.adj.sum() // 2)
        progress = float(e_now / max(1, self.m))
        # minimal rotation-invariant node features
        deg = self.adj.sum(axis=1).astype(np.float32)
        degn = deg / max(1.0, float(deg.max()))
        feats = [degn.astype(np.float32)]
        if NODE_USE_LAPPE and is_connected_adj(self.adj):
            try:
                w_small, V_small = laplacian_small_eigvecs(laplacian_from_adj(self.adj), num=3)
                if V_small.shape[1] >= 2:
                    phi2 = V_small[:, 1].astype(np.float32)
                    phi2 = phi2 - float(phi2.mean()); nrm = float(np.linalg.norm(phi2) + 1e-8); phi2 = (phi2 / nrm).astype(np.float32)
                    feats.append(phi2)
                if V_small.shape[1] >= 3:
                    phi3 = V_small[:, 2].astype(np.float32)
                    phi3 = phi3 - float(phi3.mean()); nrm = float(np.linalg.norm(phi3) + 1e-8); phi3 = (phi3 / nrm).astype(np.float32)
                    feats.append(phi3)
            except Exception:
                pass
        node_feats = np.stack(feats, axis=1)
        # global scalar features
        glob = np.array([
            math.log1p(n),
            float(k) / max(1, n - 1),
            progress,
            float(is_connected_adj(self.adj)),
            float(np.mean(np.maximum(0.0, k - deg))),  # mean deficit
        ], dtype=np.float32)
        return {
            'node_feats': node_feats,
            'global': glob,
        }

    def _components(self) -> np.ndarray:
        n = self.n
        comp = -np.ones(n, dtype=int)
        cid = 0
        A = self.adj.astype(bool)
        for i in range(n):
            if comp[i] != -1:
                continue
            stack = [i]
            comp[i] = cid
            while stack:
                u = stack.pop()
                for v in np.nonzero(A[u])[0]:
                    if comp[v] == -1:
                        comp[v] = cid
                        stack.append(int(v))
            cid += 1
        return comp

    def candidates(self, topk_frac: float = TOPK_FRAC_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        deg = self.adj.sum(axis=1)
        # deficit-only candidates (enforce regularity)
        uu, vv = np.triu_indices(n, k=1)
        mask_non = (self.adj[uu, vv] == 0)
        uu = uu[mask_non]; vv = vv[mask_non]
        if uu.size == 0:
            return uu, vv, np.zeros(0, dtype=np.float64)
        mask_def = (deg[uu] < self.k) & (deg[vv] < self.k)
        uu = uu[mask_def]; vv = vv[mask_def]
        if uu.size == 0:
            return uu, vv, np.zeros(0, dtype=np.float64)
        # connectivity-first: if disconnected, keep only cross-component edges
        comp = self._components()
        if not is_connected_adj(self.adj):
            mask_cross = (comp[uu] != comp[vv])
            # Prefer cross-component edges when available, but do not dead-end
            if np.any(mask_cross):
                uu = uu[mask_cross]; vv = vv[mask_cross]
        # Erdős–Gallai feasibility on residual deficits
        defc_full = (self.k - deg).astype(np.int64)
        if uu.size > 0:
            keep = []
            for a, b in zip(uu, vv):
                if defc_full[a] <= 0 or defc_full[b] <= 0:
                    continue
                seq = defc_full.copy()
                seq[a] -= 1; seq[b] -= 1
                if _is_graphical_erdos_gallai(seq):
                    keep.append(True)
                else:
                    keep.append(False)
            if keep:
                keep = np.array(keep, dtype=bool)
                uu = uu[keep]; vv = vv[keep]
                if uu.size == 0:
                    return uu, vv, np.zeros(0, dtype=np.float64)

        # Fiedler top-k prune after feasibility (keep ~33% by default)
        if uu.size > 0 and topk_frac is not None and 0.0 < topk_frac < 1.0:
            kcount = max(1, int(math.ceil(topk_frac * uu.size)))
            try:
                w_small, V_small = laplacian_small_eigvecs(laplacian_from_adj(self.adj), num=3)
                phi2 = V_small[:, 1] if V_small.shape[1] >= 2 else np.zeros(n)
                phi_gap = (phi2[uu] - phi2[vv]) ** 2
                idx = np.argpartition(-phi_gap, kcount - 1)[:kcount]
            except Exception:
                defc = (self.k - deg)
                score_fb = defc[uu] + defc[vv]
                idx = np.argpartition(-score_fb, kcount - 1)[:kcount]
            uu = uu[idx]; vv = vv[idx]

        # Strict feasibility mask on the pruned set (polynomial bipartite first; DFS fallback)
        if STRICT_MASK and uu.size > 0:
            cand_pairs = np.stack([uu, vv], axis=1)
            def _feas_pair(pair):
                a, b = int(pair[0]), int(pair[1])
                try:
                    H, r = _residual_host_after_add(self.adj, self.k, a, b)
                except Exception:
                    return False
                # Try bipartite f-factor per component
                if FEAS_METHOD in ('auto', 'ffactor'):
                    ok = _ffactor_bipartite_exists(H, r)
                    if ok:
                        return True
                    if FEAS_METHOD == 'ffactor':
                        return False
                # fallback to exact DFS
                return _host_constrained_completion_exists(H, r)

            if FEAS_PARALLEL and cand_pairs.shape[0] > 1:
                with _fut.ProcessPoolExecutor(max_workers=FEAS_WORKERS) as ex:
                    keep_list = list(ex.map(_feas_pair, cand_pairs))
                keep = np.array(keep_list, dtype=bool)
            else:
                keep = np.array([_feas_pair(p) for p in cand_pairs], dtype=bool)
            uu = uu[keep]; vv = vv[keep]
            if uu.size == 0:
                return uu, vv, np.zeros(0, dtype=np.float64)

        # score candidates (structural bias toward high-λ2 regular graphs)
        dens = float(self.k) / max(1, n - 1)
        # ER dependency removed: base score is deficit sum; ers zeros for features
        defc = (self.k - deg)
        base = defc[uu] + defc[vv]
        ers = np.zeros_like(base)
        score = base.copy()
        if USE_STRUCTURAL_BIAS:
            # Fiedler gap (when connected)
            if is_connected_adj(self.adj):
                try:
                    _, phi2 = lam2_smallest(laplacian_from_adj(self.adj), need_vec=True)
                    phi_gap = (phi2[uu] - phi2[vv]) ** 2
                    score = score + W_PHI * phi_gap
                except Exception:
                    pass
            # Complement φ-gap bias in mid/high density
            if dens >= COMPLEMENT_BIAS_THRESH:
                try:
                    Ac = complement_adj(self.adj)
                    _, phi2c = lam2_smallest(laplacian_from_adj(Ac), need_vec=True)
                    phi_gap_c = (phi2c[uu] - phi2c[vv]) ** 2
                    score = score + W_PHI_COMP * phi_gap_c
                except Exception:
                    pass
            # Multipartite cross-cluster bias in dense regime
            if dens >= DENSE_PART_THRESH and is_connected_adj(self.adj):
                try:
                    L = laplacian_from_adj(self.adj)
                    w_small, V_small = laplacian_small_eigvecs(L, num=3)
                    x = V_small[:, 1] if V_small.shape[1] >= 2 else None
                    y = V_small[:, 2] if V_small.shape[1] >= 3 else None
                    if x is not None:
                        if y is None:
                            lab = (x >= 0).astype(int)
                        else:
                            ang = np.arctan2(y, x)
                            r_eff = int(max(2, min(MULTI_MAX_PARTS, round(n / max(1, n - self.k)))))
                            if r_eff <= 2:
                                lab = (x >= 0).astype(int)
                            elif r_eff == 3:
                                lab = np.zeros(n, dtype=int)
                                lab[(ang >= -np.pi/3) & (ang < np.pi/3)] = 1
                                lab[ang >= np.pi/3] = 2
                                lab[ang < -np.pi/3] = 0
                            else:
                                lab = (x >= 0).astype(int) + 2 * (y >= 0).astype(int)
                        part_cross = (lab[uu] != lab[vv]).astype(np.float64)
                        score = score + W_PART * part_cross
                except Exception:
                    pass
            # Triangle control: penalize common neighbors in sparse/mid regime
            if dens <= SPARSE_TRI_THRESH:
                Au = self.adj[uu]; Av = self.adj[vv]
                overlap = (Au * Av).sum(axis=1).astype(np.float64) / max(1, n - 2)
                score = score - W_TRI_SPARSE * overlap
            # 4-cycle control: penalize edges that close many 4-cycles (sparse/mid)
            if dens <= C4_THRESH:
                try:
                    B = self.adj @ self.adj  # A^2
                    # cycles4(u,v) ≈ number of edges between N(u) and N(v) = <B[u,:], A[v,:]>
                    c4 = (B[uu] * self.adj[vv]).sum(axis=1).astype(np.float64)
                    # normalize by maximum possible pairs |N(u)|*|N(v)|
                    deg_u = deg[uu]; deg_v = deg[vv]
                    denom = np.maximum(1.0, deg_u * deg_v)
                    c4n = c4 / denom
                    score = score - W_C4_SPARSE * c4n
                except Exception:
                    pass
        # Strict feasibility mask (provable via constructive completion)
        if STRICT_MASK and uu.size > 0:
            keep = _feasible_mask_after_add(self.adj, self.k, uu, vv)
            uu = uu[keep]; vv = vv[keep]
            score = score[keep]
            if 'ers' in locals():
                ers = ers[keep]
            if uu.size == 0:
                return uu, vv, np.zeros(0, dtype=np.float64)
        # Return all feasible candidates
        if ers is None or (isinstance(ers, np.ndarray) and ers.size != uu.size):
            ers = np.zeros_like(uu, dtype=np.float64)
        return uu, vv, ers

    def step(self, u: int, v: int):
        # place edge
        if u != v and self.adj[u, v] == 0.0:
            self.adj[u, v] = 1.0; self.adj[v, u] = 1.0
            # ER dependency removed: no Green updates
        self.t += 1
        return self.state(), self.done()

    def done(self) -> bool:
        deg = self.adj.sum(axis=1)
        return (np.allclose(deg, self.k, atol=1e-6) or int(self.adj.sum() // 2) >= self.m)


# -----------------------------
# Rotation-invariant candidate features
# -----------------------------

def edge_features(env: Phase1Env, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    A = env.adj
    n, k = env.n, env.k
    deg = A.sum(axis=1).astype(np.float32)
    degn = deg / max(1.0, float(deg.max()))
    e_now = int(A.sum() // 2)
    progress = float(e_now / max(1, env.m))
    dens_target = normalized_density(n, env.m)

    # invariants
    du = degn[uu]; dv = degn[vv]
    def_u = np.maximum(0.0, (k - deg[uu])).astype(np.float32)
    def_v = np.maximum(0.0, (k - deg[vv])).astype(np.float32)
    # overlap/common neighbors
    Au = A[uu]; Av = A[vv]
    overlap = (Au * Av).sum(axis=1).astype(np.float32) / max(1, n - 2)
    # degree variance change (closed form)
    mu = 2.0 * e_now / n
    S2 = float((deg**2).sum())
    Var = S2 / n - mu * mu
    dVar = (2.0 * (deg[uu] + deg[vv]) + 2.0) / n - (4.0 * mu / n + 4.0 / (n * n))
    dVar = dVar.astype(np.float32)
    # ER disabled in training/inference features
    ers = np.zeros_like(du, dtype=np.float32)
    # cross-component indicator
    comp = env._components()
    cross = (comp[uu] != comp[vv]).astype(np.float32)

    # Structural extras for learning: set phi-gap to zero to avoid eigen cost in features
    phi_gap = np.zeros_like(du, dtype=np.float32)
    # multipartite cross via spectral sectoring (2–4 parts)
    part_cross = np.zeros_like(du, dtype=np.float32)
    part_cross = np.zeros_like(du, dtype=np.float32)
    # approximate 4-cycle density for the edge (u,v)
    try:
        B = A @ A  # A^2
        # cycles4(u,v) ~ edges between N(u) and N(v)
        c4 = (B[uu] * A[vv]).sum(axis=1).astype(np.float32)
        deg_u = deg[uu].astype(np.float32)
        deg_v = deg[vv].astype(np.float32)
        denom = np.maximum(1.0, deg_u * deg_v)
        c4n = (c4 / denom).astype(np.float32)
    except Exception:
        c4n = np.zeros_like(du, dtype=np.float32)

    edge_feats = np.stack([
        du, dv,
        np.full_like(du, progress),
        np.full_like(du, dens_target),
        cross,
        ers,
        def_u, def_v,
        dVar, overlap,
        np.full_like(du, float(np.mean(np.maximum(0.0, k - deg)))),
        phi_gap,
        part_cross,
        c4n,
    ], axis=1).astype(np.float32)
    return edge_feats


# -----------------------------
# Models
# -----------------------------

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.W = nn.Parameter(torch.randn(heads, in_dim, out_dim) * (1.0 / math.sqrt(in_dim)))
        self.a = nn.Parameter(torch.randn(heads, out_dim * 2) * 0.01)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        # add self loops
        adj = torch.maximum(adj, torch.eye(n, device=adj.device, dtype=adj.dtype))
        h = torch.einsum('hid,ni->hnd', self.W, x)  # (H, n, d)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        cat = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky(torch.einsum('hD,hndD->hnd', self.a, cat))
        mask = (adj > 0).unsqueeze(0)
        e = e.masked_fill(~mask, -1e9)
        a = torch.softmax(e, dim=2)
        out = torch.einsum('hij,hjd->hid', a, h)
        out = out.permute(1, 0, 2).reshape(n, -1)
        return out

class GATEncoder(nn.Module):
    def __init__(self, in_dim=2, hid=32, heads=4, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(GATLayer(d, hid, heads=heads))
            d = hid * heads
        self.out_dim = d

    def forward(self, x, adj):
        for g in self.layers:
            x = g(x, adj)
            x = F.elu(x)
        return x

class EdgePolicyNet(nn.Module):
    def __init__(self, node_in=1, edge_in=14, global_in=5, hid=32, heads=4, layers=2, edge_hidden=256):
        super().__init__()
        self.enc = GATEncoder(node_in, hid, heads, layers)
        # ψ-head to predict a surrogate Fiedler vector (one scalar per node)
        self.psi_head = nn.Linear(self.enc.out_dim, 1)
        # include an extra 1-dim feature for ψ-gap and optionally pairwise relational terms
        pair_mult = 4 if USE_PAIRWISE_REL else 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.enc.out_dim * pair_mult + edge_in + global_in + 1, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

    def _psi_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        psi = self.psi_head(x).squeeze(-1)  # (n,)
        # project off the constant component and normalize
        psi = psi - psi.mean()
        norm = torch.norm(psi) + 1e-8
        psi = psi / norm
        return psi

    def psi_aux_loss(self, node_feats: torch.Tensor, adj_dense: torch.Tensor) -> torch.Tensor:
        # compute embeddings and ψ, then spectral residual loss
        x = self.enc(node_feats, adj_dense)
        psi = self._psi_from_embeddings(x)
        n = adj_dense.size(0)
        deg = adj_dense.sum(dim=1)
        L = torch.diag(deg) - adj_dense
        # Rayleigh quotient and residual
        rq = torch.dot(psi, L @ psi)
        res = L @ psi - rq * psi
        res_loss = torch.mean(res * res)
        orth_loss = (psi.mean()) ** 2
        norm_loss = (torch.norm(psi) - 1.0) ** 2
        return res_loss + PSI_ORTH_W * orth_loss + PSI_NORM_W * norm_loss

    def forward(self, node_feats: torch.Tensor, adj_dense: torch.Tensor, cand_edges: torch.Tensor,
                edge_feats: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        x = self.enc(node_feats, adj_dense)
        if cand_edges.numel() == 0:
            return torch.empty(0, device=x.device)
        u = cand_edges[:, 0]; v = cand_edges[:, 1]
        xu = x[u]; xv = x[v]
        if USE_PAIRWISE_REL:
            uv = torch.cat([xu, xv, torch.abs(xu - xv), xu * xv], dim=1)
        else:
            uv = torch.cat([xu, xv], dim=1)
        gf = global_feats.unsqueeze(0).expand(uv.size(0), -1)
        # learned ψ-gap feature
        psi = self._psi_from_embeddings(x)
        psi_gap = (psi[u] - psi[v]) ** 2
        psi_gap = psi_gap.unsqueeze(-1)
        full = torch.cat([uv, edge_feats, gf, psi_gap], dim=1)
        logits = self.edge_mlp(full).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return logits


class ValueNet(nn.Module):
    def __init__(self, global_in=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feats: torch.Tensor) -> torch.Tensor:
        return self.net(global_feats).squeeze(-1)


# -----------------------------
# RL training
# -----------------------------

def _pick_k_for_bucket(n: int, bucket: int, buckets: int) -> int:
    m_min = n - 1
    m_max = n * (n - 1) // 2
    rho_mid = (bucket + 0.5) / max(1, buckets)
    m_star = m_min + rho_mid * (m_max - m_min)
    k_star = int(round(2.0 * m_star / n))
    def valid(k):
        return (3 <= k <= n - 2) and ((n * k) % 2 == 0)
    if valid(k_star):
        return k_star
    for d in range(1, n):
        for cand in (k_star - d, k_star + d):
            if valid(cand):
                return cand
    return 3 if valid(3) else (4 if valid(4) else max(3, min(n - 2, k_star)))


def sample_task_bucket(n_min: int, n_max: int, bucket: int, buckets: int) -> Tuple[int, int]:
    n = random.randint(n_min, n_max)
    k = _pick_k_for_bucket(n, bucket, buckets)
    return n, k


from typing import Optional


def final_reward(env: Phase1Env, lam2_precomputed: Optional[float] = None) -> float:
    n, k = env.n, env.k
    A = env.adj
    deg = A.sum(axis=1)
    reg = float(np.allclose(deg, k, atol=1e-6))
    conn = float(is_connected_adj(A))
    if lam2_precomputed is None:
        lam2 = lam2_smallest(laplacian_from_adj(A), need_vec=False)
    else:
        lam2 = lam2_precomputed
    return 5.0 * reg * conn + 0.3 * lam2 - 0.1 * (np.var(deg) / max(1.0, k * k))


def step_reward(env: Phase1Env, u: int, v: int) -> float:
    n, k = env.n, env.k
    A = env.adj
    deg = A.sum(axis=1)
    # regularity potential
    F = float(np.sum((np.maximum(0.0, k - deg)) ** 2))
    # simulate add
    if u == v or A[u, v] == 1.0:
        return -0.5
    deg2 = deg.copy()
    deg2[u] += 1; deg2[v] += 1
    F2 = float(np.sum((np.maximum(0.0, k - deg2)) ** 2))
    dF = F2 - F  # negative is good
    # connectivity heuristic bonus for joining components
    comp_before = _comp_count(env.adj)
    # simulate add
    adj2 = env.adj.copy(); adj2[u, v] = 1.0; adj2[v, u] = 1.0
    comp_after = _comp_count(adj2)
    bonus_conn = 0.5 * float(comp_after < comp_before)
    # spectral shaping: compute Δλ2 sparsely to save time
    compute_spectral = ((env.t % SPECTRAL_STEP_EVERY) == 0) or ((env.m - int(A.sum()//2)) <= env.n)
    if compute_spectral:
        lam2_before = lam2_smallest(laplacian_from_adj(A), need_vec=False)
        lam2_after = lam2_smallest(laplacian_from_adj(adj2), need_vec=False)
        dlam2 = lam2_after - lam2_before
        ram_margin = lam2_after - (k - 2.0 * math.sqrt(max(0.0, k - 1.0)))
    else:
        dlam2 = 0.0
        ram_margin = 0.0
    # extra gain in sparse regime
    dens = float(k) / max(1, n - 1)
    if dens <= 0.4:
        dlam2 *= (1.0 + SPARSE_STEP_LAM2_GAIN)
        ram_margin *= (1.0 + SPARSE_RAMANUJAN_GAIN)
    # complement-aware shaping for dense cases
    comp_term = 0.0
    if dens >= COMPLEMENT_BIAS_THRESH and compute_spectral:
        Ac = complement_adj(A)
        Ac2 = complement_adj(adj2)
        rc = (n - 1) - k
        lam2c_before = lam2_smallest(laplacian_from_adj(Ac), need_vec=False)
        lam2c_after = lam2_smallest(laplacian_from_adj(Ac2), need_vec=False)
        dlam2c = lam2c_after - lam2c_before
        ram_margin_c = lam2c_after - (rc - 2.0 * math.sqrt(max(0.0, rc - 1.0)))
        comp_term = COMP_STEP_LAM2_WEIGHT * dlam2c + COMP_RAMANUJAN_MARGIN_WEIGHT * ram_margin_c
    return -0.02 * dF + bonus_conn + STEP_LAM2_WEIGHT * dlam2 + RAMANUJAN_MARGIN_WEIGHT * ram_margin + comp_term

def _comp_count(A: np.ndarray) -> int:
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    A_b = A.astype(bool)
    cnt = 0
    for i in range(n):
        if seen[i]:
            continue
        cnt += 1
        stack = [i]; seen[i] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(A_b[u])[0]:
                if not seen[v]:
                    seen[v] = True; stack.append(int(v))
    return cnt


def train_phase1_rl(
    save_path: str,
    episodes: int = 5000,
    n_min: int = 8,
    n_max: int = 32,
    eval_every: int = 200,
    data_dir: str = 'data/data_ENUM',
    eval_samples: int = 20,
    device: str = 'cpu',
    seed: int = 0,
    batch_episodes: int = 16,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    save_every: int = 100,
    load_model: str = '',
    buckets: int = BUCKETS,
    use_mpi: bool = USE_MPI,
    log_every: int = 10,
    topk_frac: float = TOPK_FRAC_DEFAULT,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    # Instantiate policy with dynamic input dims from a dummy env
    tmp_env = Phase1Env(max(n_min, 3), 3 if (max(n_min,3)*3)%2==0 else 4)
    st0 = tmp_env.state()
    uu0, vv0, _ = tmp_env.candidates(topk_frac=TOPK_FRAC_DEFAULT)
    ef0 = edge_features(tmp_env, uu0[:min(8,len(uu0))], vv0[:min(8,len(vv0))])
    node_in = st0['node_feats'].shape[1]
    edge_in = ef0.shape[1]
    pol = EdgePolicyNet(node_in=node_in, edge_in=edge_in, global_in=5, hid=128, heads=8, layers=4, edge_hidden=512).to(device)
    val = ValueNet(global_in=5, hidden=128).to(device)
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()), lr=3e-4)

    # Optional resume
    if load_model:
        try:
            obj = torch.load(load_model, map_location=device)
            if isinstance(obj, dict):
                if 'policy' in obj:
                    pol.load_state_dict(obj['policy'], strict=False)
                elif 'state_dict' in obj:
                    pol.load_state_dict(obj['state_dict'], strict=False)
                if 'value' in obj:
                    val.load_state_dict(obj['value'], strict=False)
                if 'opt' in obj:
                    try:
                        opt.load_state_dict(obj['opt'])
                    except Exception:
                        pass
            print(f"[resume] Loaded weights from: {load_model}")
        except Exception as e:
            print(f"[resume] Warning: failed to load {load_model}: {e}")

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    ema_return = 0.0
    gamma = 0.99
    temp = 1.0

    buffer = []  # per-step transitions across episodes
    ep_in_batch = 0

    # MPI setup (optional)
    try:
        from mpi4py import MPI  # type: ignore
        mpi_ok = True
    except Exception:
        mpi_ok = False
    rank = 0; world = 1
    if use_mpi and mpi_ok:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank(); world = comm.Get_size()
    # select bucket per worker
    if (not use_mpi) or (use_mpi and rank == 0):
        print(f"[start] rank={rank} world={world} buckets={buckets} n∈[{n_min},{n_max}] episodes={episodes}", flush=True)

    def bucket_for_episode(ep: int) -> int:
        if use_mpi and mpi_ok:
            return int(rank % max(1, buckets))
        # single-process: cycle buckets for better generalization
        return int((ep - 1) % max(1, buckets))

    for ep in range(1, episodes + 1):
        b = bucket_for_episode(ep)
        n, k = sample_task_bucket(n_min, n_max, bucket=b, buckets=buckets)
        # avoid trivial extremes
        if k <= 1 or k >= n - 1:
            continue
        env = Phase1Env(n, k)
        traj = []  # list of (node, adj, cand, edge, glob, glob_next, act, logp_old, v_old, r, done, teacher_idx)
        done = False
        teacher_thresh = max(TEACHER_MIN_EDGES_LEFT, int(max(1, env.m) * TEACHER_EDGES_LEFT_FRAC))
        phi_thresh = max(TEACHER_MIN_EDGES_LEFT, int(max(1, env.m) * PHI_SHAPING_FRAC))
        while not done:
            st = env.state()
            glob = to_t(st['global'])
            iu, iv, _ = env.candidates(topk_frac=topk_frac)
            if iu.size == 0:
                break
            e_feats_np = edge_features(env, iu, iv)
            e_feats = to_t(e_feats_np)
            # tensors for GNN
            node_feats = to_t(env.state()['node_feats'])
            adj_dense = to_t(env.adj)
            cand = torch.as_tensor(np.stack([iu, iv], axis=1), dtype=torch.long, device=device)
            logits = pol(node_feats, adj_dense, cand, e_feats, glob) / temp
            # Guard against NaNs/Infs in logits; clamp to zeros instead of ER fallback
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            dist = Categorical(logits=logits)
            edges_left = env.m - int(env.adj.sum() // 2)
            teacher_idx = -1
            if TEACHER_FORCE and OPTIMIZE and edges_left <= teacher_thresh and iu.size > 0:
                teacher_idx = _best_lookahead_edge(env, iu, iv)
                act = torch.tensor(teacher_idx, dtype=torch.long, device=device)
            else:
                act = dist.sample()
                teacher_idx = -1
            logp_old = dist.log_prob(act).detach()
            v_old = val(glob).detach()
            a_i = int(act.detach().cpu().item())
            u = int(iu[a_i]); v = int(iv[a_i])
            # stepwise shaping without ER dependency
            r = step_reward(env, u, v)
            if PHI_SHAPING_WEIGHT > 0.0 and edges_left <= phi_thresh:
                _, phi2_vec = lam2_smallest(laplacian_from_adj(env.adj), need_vec=True)
                phi_bonus = float((phi2_vec[u] - phi2_vec[v]) ** 2)
                r += PHI_SHAPING_WEIGHT * phi_bonus
            st2, done = env.step(u, v)
            glob_next = to_t(st2['global'])
            traj.append((
                node_feats.detach(), adj_dense.detach(), cand.detach(), e_feats.detach(),
                glob.detach(), glob_next.detach(), act.detach(), logp_old, v_old, r, done, teacher_idx
            ))

        # add terminal bonus to last step reward
        Gt = 0.0
        if traj:
            lam2_pre = lam2_smallest(laplacian_from_adj(env.adj), need_vec=False)
            delta_switch = 0.0
            if SWITCH_REWARD_WEIGHT > 0.0 and np.allclose(env.adj.sum(axis=1), env.k, atol=1e-6) and is_connected_adj(env.adj):
                _, lam2_imp = _two_switch_improve_lambda2(env.adj.copy(), env.k, max_iters=50, tries_per_iter=50)
                delta_switch = max(0.0, lam2_imp - lam2_pre)
            Gt = final_reward(env, lam2_precomputed=lam2_pre) + SWITCH_REWARD_WEIGHT * delta_switch
            last = list(traj[-1])
            last[9] = last[9] + Gt  # reward index
            last[10] = True
            traj[-1] = tuple(last)

        buffer.extend(traj)
        ep_in_batch += 1

        # PPO update after batch_episodes
        if ep_in_batch >= batch_episodes and buffer:
            # build tensors
            node_list = [b[0] for b in buffer]
            adj_list = [b[1] for b in buffer]
            cand_list = [b[2] for b in buffer]
            edge_list = [b[3] for b in buffer]
            glob_list = [b[4] for b in buffer]
            globn_list = [b[5] for b in buffer]
            act_list = [b[6] for b in buffer]
            logp_old_list = [b[7] for b in buffer]
            v_old_list = [b[8] for b in buffer]
            rew_list = [b[9] for b in buffer]
            done_list = [b[10] for b in buffer]
            teacher_list = [b[11] for b in buffer]

            V_next = [val(g.to(device)).detach() for g in globn_list]
            V_old = [v for v in v_old_list]
            adv_list = []
            ret_list = []
            gae = 0.0
            for i in reversed(range(len(buffer))):
                r = rew_list[i]
                done_f = float(done_list[i])
                v = float(V_old[i].cpu().item())
                vn = float(V_next[i].cpu().item())
                delta = r + gamma * (1.0 - done_f) * vn - v
                gae = delta + gamma * 0.95 * (1.0 - done_f) * gae
                adv_list.append(gae)
                ret_list.append(gae + v)
            adv_list.reverse(); ret_list.reverse()
            adv_t = torch.as_tensor(adv_list, dtype=torch.float32, device=device)
            # normalize advantages
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
            ret_t = torch.as_tensor(ret_list, dtype=torch.float32, device=device)

            idxs = list(range(len(buffer)))
            for _ in range(ppo_epochs):
                random.shuffle(idxs)
                for j in range(0, len(idxs), 64):
                    mb = idxs[j:j+64]
                    if not mb:
                        continue
                    logp_news = []
                    ents = []
                    v_news = []
                    teach_losses = []
                    for i in mb:
                        node = node_list[i].to(device)
                        adj = adj_list[i].to(device)
                        cand = cand_list[i].to(device)
                        edgef = edge_list[i].to(device)
                        glob = glob_list[i].to(device)
                        act = act_list[i].to(device)
                        logits = pol(node, adj, cand, edgef, glob)
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                        dist = Categorical(logits=logits)
                        logp_news.append(dist.log_prob(act))
                        ents.append(dist.entropy().mean())
                        v_news.append(val(glob))
                        # ψ auxiliary spectral loss
                        teach_losses.append(PSI_AUX_WEIGHT * pol.psi_aux_loss(node, adj))
                        # k-factor teacher: build ψ-gap plan and distill toward its edges
                        if KFACTOR_TEACH:
                            with torch.no_grad():
                                x_emb = pol.enc(node, adj)
                                psi = pol._psi_from_embeddings(x_emb).detach().cpu().numpy()
                            A_np = adj.detach().cpu().numpy()
                            E_plan = _approx_kfactor_from_psi(A_np, int(round(float(glob[1].cpu().item() * (node.size(0)-1)))), psi)
                            # teacher logits over current cand: weight by ψ-gap if in plan, else small
                            cand_np = cand.detach().cpu().numpy()
                            u_np = cand_np[:,0]; v_np = cand_np[:,1]
                            psi_np = psi
                            w_list = []
                            for uu_i, vv_i in zip(u_np, v_np):
                                key = (int(min(uu_i, vv_i)), int(max(uu_i, vv_i)))
                                if key in E_plan:
                                    w_list.append(float((psi_np[uu_i] - psi_np[vv_i])**2))
                                else:
                                    w_list.append(0.0)
                            t_logits = torch.tensor(w_list, dtype=torch.float32, device=device)
                            if t_logits.numel() > 0:
                                logp = F.log_softmax(logits, dim=0)
                                p_t = F.softmax(t_logits / max(1e-6, KFACTOR_TAU), dim=0)
                                teach_losses.append(KFACTOR_TEACH_WEIGHT * (-(p_t * logp).sum()))
                        if ENUM_TEACH:
                            n_nodes = node.size(0)
                            kdeg = int(round(float(glob[1].cpu().item() * (n_nodes - 1))))
                            m_edges = n_nodes * kdeg // 2
                            E_star = _parse_enum_adj(data_dir, n_nodes, m_edges)
                            if E_star:
                                cand_np = cand.detach().cpu().numpy()
                                teach_mask = []
                                for uu_i, vv_i in zip(cand_np[:,0], cand_np[:,1]):
                                    a, b = (int(min(uu_i, vv_i)), int(max(uu_i, vv_i)))
                                    teach_mask.append(1.0 if (a, b) in E_star else 0.0)
                                t_logits2 = torch.tensor(teach_mask, dtype=torch.float32, device=device)
                                if t_logits2.sum() > 0:
                                    logp2 = F.log_softmax(logits, dim=0)
                                    p_t2 = F.softmax(t_logits2, dim=0)
                                    teach_losses.append(ENUM_TEACH_WEIGHT * (-(p_t2 * logp2).sum()))
                        # Optional structural distillation (soft labels from structural score; no lookahead)
                        if STRUCT_TEACH:
                            # edgef layout indices
                            defsum_i = edgef[:, 6] + edgef[:, 7]
                            phi_gap_i = edgef[:, 11]
                            part_cross_i = edgef[:, 12]
                            c4n_i = edgef[:, 13]
                            dens_ratio = glob[1]
                            # ER term removed from teacher
                            score_t = defsum_i + W_PHI * phi_gap_i
                            if dens_ratio >= DENSE_PART_THRESH:
                                score_t = score_t + W_PART * part_cross_i
                            if dens_ratio <= SPARSE_TRI_THRESH:
                                # triangle proxy is edgef[:,9] (overlap) already normalized
                                score_t = score_t - W_TRI_SPARSE * edgef[:, 9]
                            if dens_ratio <= C4_THRESH:
                                score_t = score_t - W_C4_SPARSE * c4n_i
                            t_logits = score_t / max(1e-6, STRUCT_TEACH_TAU)
                            # Cross-entropy with soft targets
                            logp = F.log_softmax(logits, dim=0)
                            p_t = F.softmax(t_logits, dim=0)
                            teach_losses.append(-(p_t * logp).sum())
                        # Retain optional end-game single-index teacher if enabled (still off unless OPTIMIZE)
                        teacher_idx = teacher_list[i]
                        if TEACHER_FORCE and teacher_idx is not None and teacher_idx >= 0:
                            target = torch.tensor([teacher_idx], dtype=torch.long, device=device)
                            teach_losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                    logp_news = torch.stack(logp_news)
                    ents = torch.stack(ents)
                    v_news = torch.stack(v_news)
                    if v_news.dim() == 0:
                        v_news = v_news.unsqueeze(0)
                    v_news = v_news.view(-1)
                    logp_old_mb = torch.stack([logp_old_list[i] for i in mb]).to(device)
                    adv_mb = adv_t[mb]
                    ret_mb = ret_t[mb]
                    ratio = torch.exp(logp_news - logp_old_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                    pol_loss = -torch.mean(torch.min(surr1, surr2)) - ent_coef * ents.mean()
                    if teach_losses:
                        teacher_loss = torch.stack(teach_losses).mean()
                        # Combine structured teacher and optional end-game teacher via weights
                        pol_loss = pol_loss + (STRUCT_TEACH_WEIGHT + TEACHER_KL_WEIGHT) * teacher_loss
                    v_loss = vf_coef * F.mse_loss(v_news, ret_mb)
                    loss = pol_loss + v_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + list(val.parameters()), max_norm=0.5)
                    opt.step()

            buffer.clear(); ep_in_batch = 0

        ema_return = 0.95 * ema_return + 0.05 * float(Gt)
        if log_every > 0 and (ep % log_every) == 0:
            # In MPI, print from rank 0 to avoid clutter
            if (not use_mpi) or (use_mpi and (rank == 0)):
                print(f"[ep {ep}] rank={rank} bucket={b}/{buckets} n={n} k={k} steps={len(traj)} Gt={Gt:.3f} emaR={ema_return:.3f}", flush=True)

        if eval_every > 0 and (ep % eval_every) == 0:
            eval_on_enum(pol, n_min, n_max, data_dir, eval_samples, device, topk_frac=args.topk_frac)

        # periodic checkpointing
        if save_every > 0 and (ep % save_every) == 0:
            base = os.path.splitext(save_path)[0]
            suffix = f"_b{b}_r{rank}" if (use_mpi and mpi_ok) else ""
            ckpt_path = f"{base}{suffix}_ep{ep:05d}.pt"
            try:
                torch.save({'policy': pol.state_dict(), 'value': val.state_dict(), 'opt': opt.state_dict()}, ckpt_path)
                print(f"[save] -> {ckpt_path}")
            except Exception as e:
                print(f"[save] Warning: failed to save checkpoint {ckpt_path}: {e}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({'policy': pol.state_dict(), 'value': val.state_dict(), 'opt': opt.state_dict()}, save_path)
    print(f"[phase1-rl] saved -> {save_path}")

    # Optional: bundle all bucket-specific models into a single file on rank 0
    if use_mpi and 'mpi_ok' in locals() and mpi_ok:
        try:
            states = comm.gather({'bucket': b, 'rank': rank,
                                  'policy': pol.state_dict(), 'value': val.state_dict()}, root=0)
            if rank == 0:
                bundled = {
                    'buckets': buckets,
                    'world_size': world,
                    'models': {f"bucket_{s['bucket']}": {'policy': s['policy'], 'value': s['value'], 'rank': s['rank']}
                               for s in states},
                }
                base = os.path.splitext(save_path)[0]
                bundle_path = f"{base}_bundle.pt"
                torch.save(bundled, bundle_path)
                print(f"[phase1-rl] bundled -> {bundle_path}")
        except Exception as e:
            if rank == 0:
                print(f"[phase1-rl] Warning: bundle save failed: {e}")


def load_enum_regular_cases(data_dir: str) -> List[Tuple[int, int]]:
    out = []
    pat = re.compile(r"data_(\d+)_(\d+)\.csv$")
    for fname in os.listdir(data_dir):
        mobj = pat.match(fname)
        if not mobj:
            continue
        n = int(mobj.group(1)); m = int(mobj.group(2))
        if (2 * m) % n != 0 or m == n or m == n * (n - 1) // 2:
            continue
        out.append((n, m))
    return out


@torch.no_grad()
def _parse_enum_lambda(data_dir: str, n: int, m: int) -> float:
    """Parse the first float in data_{n}_{m}.csv as target lambda2, if present."""
    path = os.path.join(data_dir, f"data_{n}_{m}.csv")
    try:
        with open(path, 'r') as f:
            line = f.readline().strip()
            return float(line)
    except Exception:
        return float('nan')


def _parse_enum_adj(data_dir: str, n: int, m: int) -> set[tuple[int, int]]:
    """Parse adjacency from data_{n}_{m}.csv and return edge set {(u,v): u<v}.
    Returns empty set if file missing or malformed.
    """
    path = os.path.join(data_dir, f"data_{n}_{m}.csv")
    E = set()
    try:
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # skip first line (lambda)
        for ln in lines[1:]:
            if ':' not in ln:
                continue
            left, right = ln.split(':', 1)
            u = int(left.strip())
            nbrs = right.strip()
            if not nbrs:
                continue
            if nbrs.startswith(','):
                nbrs = nbrs[1:]
            nbrs = nbrs.replace(' ', '')
            if nbrs:
                for tok in nbrs.split(','):
                    if tok == '':
                        continue
                    v = int(tok)
                    a, b = (u, v) if u < v else (v, u)
                    if a != b:
                        E.add((a, b))
    except Exception:
        return set()
    return E


@torch.no_grad()
def eval_on_enum(pol: EdgePolicyNet, n_min: int, n_max: int, data_dir: str, samples: int, device: str, topk_frac: float = TOPK_FRAC_DEFAULT):
    cases = load_enum_regular_cases(data_dir)
    if not cases:
        print('[eval] no regular cases found')
        return
    sample = random.sample(cases, min(samples, len(cases)))
    success = 0; lam2s = []
    fit_total = 0; fit_on_regular = 0; regular_count = 0
    for (n, m) in sample:
        k = (2 * m) // n
        lam2_enum = _parse_enum_lambda(data_dir, n, m)
        env = Phase1Env(n, k)
        steps = 0
        while not env.done() and steps < env.m + n:
            st = env.state()
            glob = torch.as_tensor(st['global'], dtype=torch.float32, device=device)
            iu, iv, _ = env.candidates(topk_frac=topk_frac)
            if iu.size == 0:
                break
            e_feats_np = edge_features(env, iu, iv)
            e_feats = torch.as_tensor(e_feats_np, dtype=torch.float32, device=device)
            node_feats = torch.as_tensor(st['node_feats'], dtype=torch.float32, device=device)
            adj_dense = torch.as_tensor(env.adj, dtype=torch.float32, device=device)
            cand = torch.as_tensor(np.stack([iu, iv], axis=1), dtype=torch.long, device=device)
            logits = pol(node_feats, adj_dense, cand, e_feats, glob)
            a = int(torch.argmax(logits).detach().cpu().item())
            env.step(int(iu[a]), int(iv[a]))
            steps += 1
        deg = env.adj.sum(axis=1)
        is_reg = np.allclose(deg, k, atol=1e-6) and is_connected_adj(env.adj)
        if is_reg:
            success += 1; regular_count += 1
        lam2_val = lam2_smallest(laplacian_from_adj(env.adj), need_vec=False)
        lam2s.append(lam2_val)
        if not math.isnan(lam2_enum):
            if abs(lam2_val - lam2_enum) < 1e-6:
                fit_total += 1
                if is_reg:
                    fit_on_regular += 1
    rate = success / max(1, len(sample))
    fit_rate = fit_total / max(1, len(sample))
    fit_reg_rate = fit_on_regular / max(1, regular_count) if regular_count>0 else 0.0
    print(f"[eval] regular_success={rate:.2f} median_lam2={np.median(lam2s):.3f} lam2_fit={fit_rate:.2f} lam2_fit_on_regular={fit_reg_rate:.2f} on {len(sample)} cases")


@torch.no_grad()
def infer_regular(load_path: str, n: int, m: int, device: str = 'cpu', topk_frac: float = TOPK_FRAC_DEFAULT):
    if (2 * m) % n != 0:
        print(f"[infer] (n={n}, m={m}) not regularizable: 2m % n != 0")
        return
    k = (2 * m) // n
    t0 = time.time()
    env = Phase1Env(n, k)
    # instantiate with dynamic dims from env
    st0 = env.state()
    iu0, iv0, _ = env.candidates(topk_frac=topk_frac)
    ef0 = edge_features(env, iu0[:min(8,len(iu0))], iv0[:min(8,len(iv0))])
    node_in = st0['node_feats'].shape[1]
    edge_in = ef0.shape[1]
    pol = EdgePolicyNet(node_in=node_in, edge_in=edge_in, global_in=5, hid=128, heads=8, layers=4, edge_hidden=512).to(device)
    obj = torch.load(load_path, map_location=device)
    try:
        pol.load_state_dict(obj['policy'] if 'policy' in obj else obj, strict=False)
    except Exception:
        # fallback strict=False for older checkpoints
        sd = obj['policy'] if isinstance(obj, dict) and 'policy' in obj else obj
        pol.load_state_dict(sd, strict=False)
    pol.eval()
    steps = 0
    while not env.done() and steps < env.m + n:
        st = env.state()
        glob = torch.as_tensor(st['global'], dtype=torch.float32, device=device)
        iu, iv, _ = env.candidates(topk_frac=topk_frac)
        if iu.size == 0:
            break
        e_feats_np = edge_features(env, iu, iv)
        e_feats = torch.as_tensor(e_feats_np, dtype=torch.float32, device=device)
        node_feats = torch.as_tensor(st['node_feats'], dtype=torch.float32, device=device)
        adj_dense = torch.as_tensor(env.adj, dtype=torch.float32, device=device)
        cand = torch.as_tensor(np.stack([iu, iv], axis=1), dtype=torch.long, device=device)
        logits = pol(node_feats, adj_dense, cand, e_feats, glob)
        a = int(torch.argmax(logits).detach().cpu().item())
        env.step(int(iu[a]), int(iv[a]))
        steps += 1
    deg = env.adj.sum(axis=1)
    is_reg = np.allclose(deg, k, atol=1e-6) and is_connected_adj(env.adj)
    lam2 = lam2_smallest(laplacian_from_adj(env.adj), need_vec=False)
    runtime = time.time() - t0
    print(f"[infer] n={n} m={m} k={k} regular={is_reg} lam2={lam2:.6f} runtime={runtime:.2f}s")
    if not is_reg:
        for i in range(n):
            nbrs = [str(j) for j in range(n) if env.adj[i, j] > 0.5]
            print(f"{i}: "+', '.join(nbrs))




def _best_lookahead_edge(env: Phase1Env, iu: np.ndarray, iv: np.ndarray) -> int:
    best_idx = 0
    best_lam2 = -1e9
    for ci in range(len(iu)):
        u = int(iu[ci]); v = int(iv[ci])
        lam2_val = _lam2_after_add_and_regular_fill(env, u, v)
        if lam2_val > best_lam2:
            best_lam2 = lam2_val; best_idx = ci
    return best_idx

def _lam2_after_add_and_regular_fill(env: Phase1Env, u: int, v: int) -> float:
    # copy state
    A = env.adj.copy(); n = env.n; k = env.k
    if u != v and A[u, v] == 0.0:
        A[u, v] = 1.0; A[v, u] = 1.0
    # init Green if connected
    use_green = False; G = None; dG = None
    if is_connected_adj(A):
        try:
            G, dG = G_from_adj(A); use_green = True
        except Exception:
            use_green = False
    # greedy fill deficit pairs by ER until regular
    while True:
        deg = A.sum(axis=1)
        if np.allclose(deg, k, atol=1e-6):
            break
        uu, vv = np.triu_indices(n, k=1)
        mask_non = (A[uu, vv] == 0)
        uu = uu[mask_non]; vv = vv[mask_non]
        if uu.size == 0:
            break
        mask_def = (deg[uu] < k) & (deg[vv] < k)
        uu = uu[mask_def]; vv = vv[mask_def]
        if uu.size == 0:
            break
        if use_green and (G is not None):
            ers = (dG[uu] + dG[vv] - 2.0 * G[uu, vv])
            idx = int(np.argmax(ers))
        else:
            # fallback to deficit sum
            defc = (k - deg)
            score = defc[uu] + defc[vv]
            idx = int(np.argmax(score))
        a = int(uu[idx]); b = int(vv[idx])
        A[a, b] = 1.0; A[b, a] = 1.0
        if use_green and (G is not None):
            try:
                G, dG = G_rank1_update_edge(G, dG, a, b, 1.0)
            except Exception:
                use_green = False
        else:
            if is_connected_adj(A):
                try:
                    G, dG = G_from_adj(A); use_green = True
                except Exception:
                    use_green = False
    w, _ = np.linalg.eigh(laplacian_from_adj(A))
    w = np.sort(w); return float(w[1]) if len(w) > 1 else 0.0


def _two_switch_improve_lambda2(A: np.ndarray, k: int, max_iters: int = 200, tries_per_iter: int = 200) -> Tuple[np.ndarray, float]:
    """Hill-climb λ2 with degree-preserving 2-switches:
    pick edges (u,a),(v,b) and swap to (u,v),(a,b) if simple/connected and λ2 improves.
    Returns improved adjacency and λ2.
    """
    n = A.shape[0]
    def lam2_of(X):
        return lam2_smallest(laplacian_from_adj(X), need_vec=False)
    lam2_cur = lam2_of(A)
    for it in range(max_iters):
        improved = False
        # compute Fiedler vector for guidance
        lam2_tmp, phi2 = lam2_smallest(laplacian_from_adj(A), need_vec=True)
        # candidate non-edges sorted by phi-gap
        uu, vv = np.triu_indices(n, k=1)
        mask_non = (A[uu, vv] == 0)
        uu = uu[mask_non]; vv = vv[mask_non]
        if uu.size == 0:
            break
        gaps = (phi2[uu] - phi2[vv]) ** 2
        order = np.argsort(-gaps)
        cand_pairs = list(zip(uu[order], vv[order]))
        # try a limited number of attempts
        tries = 0
        for (u, v) in cand_pairs:
            if tries >= tries_per_iter:
                break
            # select neighbors for swap
            Nu = np.nonzero(A[u])[0]; Nv = np.nonzero(A[v])[0]
            if Nu.size == 0 or Nv.size == 0:
                tries += 1; continue
            # try a few neighbor pairs
            for a in Nu[:min(len(Nu), 8)]:
                if a == v: continue
                for b in Nv[:min(len(Nv), 8)]:
                    if b == u or a == b: continue
                    # candidate new edges (u,v) and (a,b), old edges (u,a),(v,b)
                    if A[u, v] == 0 and A[a, b] == 0 and A[u, a] == 1 and A[v, b] == 1 and A[u, b] == 0 and A[v, a] == 0:
                        A2 = A.copy()
                        # remove
                        A2[u, a] = A2[a, u] = 0.0
                        A2[v, b] = A2[b, v] = 0.0
                        # add
                        A2[u, v] = A2[v, u] = 1.0
                        A2[a, b] = A2[b, a] = 1.0
                        # check connectivity and degrees
                        if not is_connected_adj(A2):
                            continue
                        deg2 = A2.sum(axis=1)
                        if not np.allclose(deg2, k, atol=1e-6):
                            continue
                        lam2_new = lam2_of(A2)
                        if lam2_new > lam2_cur + 1e-9:
                            A = A2; lam2_cur = lam2_new; improved = True; break
                if improved:
                    break
            tries += 1
        if not improved:
            break
    return A, lam2_cur


def _circulant_k_regular(n: int, k: int) -> np.ndarray:
    """Deterministic k-regular simple graph on n nodes (exists when n*k is even).
    Uses circulant connections: connect i to i±1..±⌊k/2⌋, and if k is odd, also to i + n/2 (n must be even).
    """
    assert 0 <= k < n, 'k must be in [0,n)'
    assert (n * k) % 2 == 0, 'n*k must be even for a k-regular simple graph'
    A = np.zeros((n, n), dtype=np.float64)
    half = k // 2
    for i in range(n):
        for s in range(1, half + 1):
            j1 = (i + s) % n
            j2 = (i - s) % n
            A[i, j1] = A[j1, i] = 1.0
            A[i, j2] = A[j2, i] = 1.0
        if (k % 2) == 1:
            assert (n % 2) == 0, 'k odd requires n even'
            j = (i + n // 2) % n
            A[i, j] = A[j, i] = 1.0
    return A


def _complete_regular_from_partial(A_in: np.ndarray, k: int) -> np.ndarray:
    """Try to complete a partial simple graph A_in to exactly k-regular by greedy EG-feasible additions.
    If it fails to find any feasible addition but degrees are still below k, fall back to circulant k-regular.
    Returns a k-regular adjacency (not necessarily preserving A_in if fallback is used).
    """
    A = A_in.copy()
    n = A.shape[0]
    # Quick path: already regular
    deg = A.sum(axis=1)
    if np.allclose(deg, k, atol=1e-6):
        return A
    # Optional: initialize Green function when connected for ER scoring
    use_green = False; G = None; dG = None
    if is_connected_adj(A):
        try:
            G, dG = G_from_adj(A); use_green = True
        except Exception:
            use_green = False
    # Greedy EG-feasible fill
    max_iters = n * n
    it = 0
    while it < max_iters:
        it += 1
        deg = A.sum(axis=1)
        if np.allclose(deg, k, atol=1e-6):
            return A
        defc_full = (k - deg).astype(np.int64)
        uu, vv = np.triu_indices(n, k=1)
        mask_non = (A[uu, vv] == 0)
        uu = uu[mask_non]; vv = vv[mask_non]
        if uu.size == 0:
            break
        mask_def = (deg[uu] < k) & (deg[vv] < k)
        uu = uu[mask_def]; vv = vv[mask_def]
        if uu.size == 0:
            break
        # EG-feasible filter
        keep = []
        for a, b in zip(uu, vv):
            seq = defc_full.copy();
            if seq[a] <= 0 or seq[b] <= 0:
                keep.append(False); continue
            seq[a] -= 1; seq[b] -= 1
            keep.append(_is_graphical_erdos_gallai(seq))
        if keep:
            keep = np.array(keep, dtype=bool)
            uu = uu[keep]; vv = vv[keep]
        if uu.size == 0:
            break
        # Score candidates: prefer ER if available, else deficit sum
        if use_green and (G is not None):
            ers = (dG[uu] + dG[vv] - 2.0 * G[uu, vv])
            idx = int(np.argmax(ers))
        else:
            defc = (k - deg)
            score = defc[uu] + defc[vv]
            idx = int(np.argmax(score))
        a = int(uu[idx]); b = int(vv[idx])
        # add edge and update ER state if possible
        A[a, b] = 1.0; A[b, a] = 1.0
        if use_green and (G is not None):
            try:
                G, dG = G_rank1_update_edge(G, dG, a, b, 1.0)
            except Exception:
                use_green = False
        else:
            if is_connected_adj(A):
                try:
                    G, dG = G_from_adj(A); use_green = True
                except Exception:
                    use_green = False
    # Greedy failed: fall back to a canonical k-regular circulant
    try:
        return _circulant_k_regular(n, k)
    except AssertionError:
        # As a last safety check, if impossible (should not happen when n*k even), return input
        return A_in


def _feasible_mask_after_add(A: np.ndarray, k: int, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """For each candidate (u,v), decide if adding it still allows a completion to k-regular
    using a constructive exact search (host‑constrained Havel–Hakimi with backtracking).

    Returns a boolean mask of shape (len(uu),): True iff a completion exists.
    This mask has no false positives; it may be conservative if search prunes too hard.
    """
    n = A.shape[0]
    mask = np.zeros(len(uu), dtype=bool)
    # Precompute host availability matrix (non-edges)
    H0 = (A == 0).astype(np.uint8)
    np.fill_diagonal(H0, 0)
    deg = A.sum(axis=1)
    for i, (u, v) in enumerate(zip(uu, vv)):
        if u == v or A[u, v] == 1.0:
            mask[i] = False; continue
        # residual after adding (u,v)
        r = (k - deg).astype(np.int64)
        if r[u] <= 0 or r[v] <= 0:
            mask[i] = False; continue
        r[u] -= 1; r[v] -= 1
        if np.any(r < 0) or (np.sum(r) % 2 != 0):
            mask[i] = False; continue
        # host after adding: forbid (u,v)
        H = H0.copy()
        H[u, v] = H[v, u] = 0
        # Quick necessary capacity check: r_i <= available neighbors with positive r
        pos = (r > 0)
        if pos.any():
            avail = (H[pos][:, pos]).sum(axis=1)
            if np.any(r[pos] > avail):
                mask[i] = False; continue
        # Try constructive completion
        ok = _host_constrained_completion_exists(H, r)
        mask[i] = ok
    return mask


def _host_constrained_completion_exists(H: np.ndarray, r: np.ndarray) -> bool:
    """Return True iff there exists a simple graph F ⊆ H whose degree sequence equals r.
    Complete, backtracking search with strong pruning; guarantees correctness.
    H: 0/1 symmetric matrix without self-loops. r: integer residual degrees ≥ 0.
    """
    n = H.shape[0]
    r = r.copy().astype(np.int64)
    # Trivial success
    if np.all(r == 0):
        return True
    # Quick necessary checks
    if np.any(r < 0):
        return False
    if (np.sum(r) % 2) != 0:
        return False
    pos_idx = np.where(r > 0)[0]
    if pos_idx.size == 0:
        return True
    # capacity per node restricted to positive residuals
    avail = (H[pos_idx][:, pos_idx]).sum(axis=1)
    if np.any(r[pos_idx] > avail):
        return False
    # Sort working set by decreasing residual (tie-break by availability)
    order = sorted(pos_idx.tolist(), key=lambda i: (int(r[i]), int(avail[np.where(pos_idx==i)[0][0]])), reverse=True)

    # Recursive backtracking with HH-style choice
    used = np.zeros_like(H, dtype=np.uint8)

    def necessary(r_local: np.ndarray) -> bool:
        # sum even and capacity
        if (np.sum(r_local) % 2) != 0:
            return False
        idx = np.where(r_local > 0)[0]
        if idx.size == 0:
            return True
        avail_local = (H[idx][:, idx] & (used[idx][:, idx] == 0)).sum(axis=1)
        if np.any(r_local[idx] > avail_local):
            return False
        # Erdos-Gallai necessary (ignoring host) to prune
        if not _is_graphical_erdos_gallai(r_local):
            return False
        return True

    def dfs(r_local: np.ndarray) -> bool:
        # done?
        if np.all(r_local == 0):
            return True
        # choose vertex with largest residual
        I = np.where(r_local > 0)[0]
        # choose with max r, tie-break by available degree
        best = None; best_key = (-1, -1)
        for i in I:
            avail_i = int(((H[i] == 1) & (used[i] == 0) & (r_local > 0)).sum())
            key = (int(r_local[i]), avail_i)
            if key > best_key:
                best = i; best_key = key
        i = int(best)
        need = int(r_local[i])
        # neighbor candidates
        J = np.where((H[i] == 1) & (used[i] == 0) & (r_local > 0))[0].tolist()
        if len(J) < need:
            return False
        # Heuristic ordering: high residual first
        J.sort(key=lambda j: int(r_local[j]), reverse=True)
        # Place edges incident to i one by one
        # Backtracking with pruning after each placement
        # Use iterative deepening over positions
        def try_place(pos: int, placed: int) -> bool:
            if placed == need:
                # i satisfied; block i from further edges
                saved_ri = int(r_local[i])
                r_local[i] = 0
                # Mark i as used with everyone to prevent accidental reuse; not strictly needed
                # but reduces branching. We only need to ensure we don't add more edges to i.
                return dfs(r_local)
            # dynamic candidate list shrinks as neighbors exhaust residual
            for t in range(pos, len(J)):
                j = int(J[t])
                if r_local[j] <= 0:
                    continue
                if used[i, j] == 1 or H[i, j] == 0:
                    continue
                # add (i,j)
                used[i, j] = used[j, i] = 1
                r_local[i] -= 1; r_local[j] -= 1
                ok = False
                if necessary(r_local):
                    ok = try_place(t + 1, placed + 1)
                # undo
                r_local[i] += 1; r_local[j] += 1
                used[i, j] = used[j, i] = 0
                if ok:
                    return True
            return False

        return try_place(0, 0)

    return dfs(r.copy())


# -----------------------------
# Polynomial-time feasibility on bipartite components
# -----------------------------

def _residual_host_after_add(A: np.ndarray, k: int, u: int, v: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, r) after hypothetically adding edge (u,v) to A.
    H is 0/1 non-edge host matrix; r integer residual degrees >=0.
    Raises ValueError if edge invalid or immediate infeasible.
    """
    n = A.shape[0]
    if u == v or A[u, v] == 1.0:
        raise ValueError('invalid edge')
    deg = A.sum(axis=1)
    r = (k - deg).astype(np.int64)
    if r[u] <= 0 or r[v] <= 0:
        raise ValueError('no residual capacity')
    r[u] -= 1; r[v] -= 1
    if np.any(r < 0) or (np.sum(r) % 2 != 0):
        raise ValueError('immediate infeasible')
    H = (A == 0).astype(np.uint8)
    np.fill_diagonal(H, 0)
    H[u, v] = H[v, u] = 0
    return H, r


def _is_bipartite_vertices(H: np.ndarray, idx: np.ndarray) -> tuple[bool, np.ndarray]:
    """Check bipartiteness of induced subgraph H[idx][:,idx]. Returns (ok, color)
    color is an array of size len(idx) with values in {0,1} or -1 for unused.
    """
    m = len(idx)
    pos = {int(idx[i]): i for i in range(m)}
    color = -np.ones(m, dtype=np.int8)
    for i0 in range(m):
        if color[i0] != -1:
            continue
        color[i0] = 0
        q = [i0]
        while q:
            qi = q.pop()
            vi = idx[qi]
            nbrs = np.nonzero(H[vi])[0]
            for w in nbrs:
                if w not in pos:
                    continue
                j = pos[w]
                if color[j] == -1:
                    color[j] = 1 - color[qi]
                    q.append(j)
                elif color[j] == color[qi]:
                    return False, color
    return True, color


class _Dinic:
    def __init__(self, N: int):
        self.N = N
        self.adj = [[] for _ in range(N)]

    def add_edge(self, u: int, v: int, cap: int):
        self.adj[u].append([v, cap, None])
        self.adj[v].append([u, 0, None])
        self.adj[u][-1][2] = self.adj[v][-1]
        self.adj[v][-1][2] = self.adj[u][-1]

    def maxflow(self, s: int, t: int) -> int:
        flow = 0
        N = self.N
        while True:
            level = [-1] * N
            q = [s]; level[s] = 0
            for u in q:
                for v, cap, rev in self.adj[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[t] < 0:
                break
            it = [0] * N

            def dfs(u, f):
                if u == t:
                    return f
                for i in range(it[u], len(self.adj[u])):
                    it[u] = i
                    v, cap, rev = self.adj[u][i]
                    if cap > 0 and level[u] + 1 == level[v]:
                        d = dfs(v, min(f, cap))
                        if d > 0:
                            self.adj[u][i][1] -= d
                            rev[1] += d
                            return d
                return 0

            while True:
                pushed = dfs(s, 10**9)
                if pushed == 0:
                    break
                flow += pushed
        return flow


def _ffactor_bipartite_exists(H: np.ndarray, r: np.ndarray) -> bool:
    """Check f-factor existence on each connected component that is bipartite via max-flow.
    Returns True if every nontrivial component admits a bipartite b-matching satisfying r.
    Falls back to True on trivial (r==0) components. Caller handles non-bipartite case.
    """
    n = H.shape[0]
    # Work on the subgraph induced by vertices with r>0
    active = np.where(r > 0)[0]
    if active.size == 0:
        return True
    visited = np.zeros(n, dtype=bool)

    for src in active:
        if visited[src]:
            continue
        # BFS to collect component vertices intersecting active
        comp = []
        q = [src]; visited[src] = True
        while q:
            u = q.pop()
            comp.append(u)
            for v in np.nonzero(H[u])[0]:
                if not visited[v] and (r[v] > 0 or True):  # include all neighbors; r[v] may be zero
                    visited[v] = True
                    q.append(int(v))
        comp = np.array(comp, dtype=int)
        # Restrict to vertices with r>0 inside component
        comp_rpos = comp[r[comp] > 0]
        if comp_rpos.size == 0:
            continue
        ok, color = _is_bipartite_vertices(H, comp)
        if not ok:
            return False  # non-bipartite component; caller should fallback
        # Build flow on bipartite graph for this component
        # Map comp vertices to side U (color 0) and V (color 1)
        pos = {int(comp[i]): i for i in range(len(comp))}
        U_idx = [i for i, c in enumerate(color) if c == 0]
        V_idx = [i for i, c in enumerate(color) if c == 1]
        # If either side empty, handle trivial
        # Build network: s->U caps r[u], U->V caps (1 for edges in H), V->t caps r[v]
        s = 0
        # node indexing: s(0), U nodes 1..|U|, V nodes |U|+1..|U|+|V|, t = last
        offset_U = 1
        offset_V = offset_U + len(U_idx)
        t = offset_V + len(V_idx)
        din = _Dinic(t + 1)
        total_demand = 0
        for i in U_idx:
            v = int(comp[i])
            cap = int(max(0, r[v]))
            if cap > 0:
                din.add_edge(s, offset_U + U_idx.index(i), cap)
                total_demand += cap
        # edges U->V
        for i in U_idx:
            u = int(comp[i])
            ui = offset_U + U_idx.index(i)
            for w in np.nonzero(H[u])[0]:
                if w not in pos:
                    continue
                j = pos[w]
                if color[j] != 1:
                    continue
                vj = offset_V + V_idx.index(j)
                din.add_edge(ui, vj, 1)
        # V->t caps
        for j in V_idx:
            v = int(comp[j])
            cap = int(max(0, r[v]))
            if cap > 0:
                din.add_edge(offset_V + V_idx.index(j), t, cap)
        flow = din.maxflow(s, t)
        if flow < total_demand:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description='Phase‑1 RL for regular graph construction (independent)')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--episodes', type=int, default=5000)
    ap.add_argument('--n_min', type=int, default=8)
    ap.add_argument('--n_max', type=int, default=32)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--data_dir', type=str, default='data/data_ENUM')
    ap.add_argument('--eval_samples', type=int, default=20)
    ap.add_argument('--save_model', type=str, default='models_phase1/p1.pt')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8, help='Set OMP/MKL/OPENBLAS threads and torch.set_num_threads')
    ap.add_argument('--ppo_mbsize', type=int, default=64)
    ap.add_argument('--refine_iters', type=int, default=100)
    ap.add_argument('--refine_tries', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=100, help='Save checkpoint every N episodes (0 to disable)')
    ap.add_argument('--buckets', type=int, default=BUCKETS)
    ap.add_argument('--mpi', action='store_true')
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--topk_frac', type=float, default=TOPK_FRAC_DEFAULT)

    # inference
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--load_model', type=str, default='')
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--m', type=int, default=0)
    args = ap.parse_args()

    # thread control
    if args.threads and args.threads > 0:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['MKL_NUM_THREADS'] = str(args.threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
        try:
            torch.set_num_threads(args.threads)
        except Exception:
            pass

    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)

    if args.train:
        train_phase1_rl(
            save_path=args.save_model,
            episodes=args.episodes,
            n_min=args.n_min,
            n_max=args.n_max,
            eval_every=args.eval_every,
            data_dir=args.data_dir,
            eval_samples=args.eval_samples,
            device=args.device,
            seed=args.seed,
            batch_episodes=16,
            ppo_epochs=4,
            clip_eps=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            save_every=args.save_every,
            load_model=args.load_model,
            buckets=args.buckets,
            use_mpi=args.mpi,
            log_every=args.log_every,
        )
    elif args.infer:
        if not args.load_model or args.n <= 0 or args.m <= 0:
            raise SystemExit('--infer requires --load_model, --n, --m')
        infer_regular(args.load_model, args.n, args.m, device=args.device, topk_frac=args.topk_frac)
    else:
        print('Specify --train or --infer')


if __name__ == '__main__':
    main()
