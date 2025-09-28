import os, sys, math, time, json, random, argparse
import csv
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    _HAVE_SCIPY=True
except Exception:
    _HAVE_SCIPY=False

# -----------------------------
# Utilities
# -----------------------------

def set_num_threads(threads: int):
    if threads and threads>0:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
        torch.set_num_threads(threads)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# episodic save path helper and simple CSV logger
def episodic_path(path: str, ep: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_ep{ep:05d}{ext}"


def log_row(csv_path: str, header: list, row: list):
    if not csv_path:
        return
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)

def log_jsonl(path: str, record: dict):
    if not path:
        return
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
    except Exception:
        pass

# -----------------------------
# Spectral helpers (CPU-focused; tiny GPU path via torch if requested)
# -----------------------------

def laplacian_from_adj(adj: np.ndarray) -> np.ndarray:
    # adj: (n,n) symmetric 0/1, zero diag
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    return L

def is_connected_adj(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    if n == 0:
        return True
    # quick check: at least n-1 edges
    if int(adj.sum() // 2) < n - 1:
        return False
    # BFS/DFS from node 0
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
    # Full eigen + pseudoinverse (zero eigen(s) -> 0)
    w, V = np.linalg.eigh(L)
    tol = 1e-12
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    G = (V * invw) @ V.T
    # symmetrize
    G = 0.5 * (G + G.T)
    return G

def G_from_adj(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    G = laplacian_pseudoinverse(laplacian_from_adj(adj))
    diagG = np.diag(G).copy()
    return G, diagG

def er_topk_fast_from_G(adj: np.ndarray, G: np.ndarray, diagG: np.ndarray, k: int, frac_cap: float):
    n = adj.shape[0]
    iu, iv = np.triu_indices(n, k=1)
    mask_non = (adj[iu, iv] == 0)
    iu = iu[mask_non]; iv = iv[mask_non]
    if iu.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    er = diagG[iu] + diagG[iv] - 2.0 * G[iu, iv]
    total = len(er)
    if frac_cap > 0:
        cap = int(math.ceil(frac_cap * total))
        k_eff = min(k, cap)
    else:
        k_eff = k
    kk = min(k_eff, total)
    if kk <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    idx = np.argpartition(-er, kk-1)[:kk]
    order = np.argsort(-er[idx])
    idx = idx[order]
    return iu[idx], iv[idx], er[idx]

def er_pairs_from_G(G: np.ndarray, diagG: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    u = pairs[:,0]; v = pairs[:,1]
    return (diagG[u] + diagG[v] - 2.0 * G[u, v]).astype(np.float64)

def G_rank1_update_edge(G: np.ndarray, diagG: np.ndarray, u: int, v: int, w: float = 1.0):
    # b = e_u - e_v; g = G b
    g = G[:, u] - G[:, v]
    # denom = 1/w + R_uv
    R_uv = float(G[u, u] + G[v, v] - 2.0 * G[u, v])
    denom = (1.0 / w) + R_uv
    if denom <= 0:
        return G, diagG  # should not happen; be safe
    # rank-one update
    outer = np.outer(g, g) / denom
    G2 = G - outer
    G2 = 0.5 * (G2 + G2.T)
    diagG2 = diagG - (g * g) / denom
    return G2, diagG2

# Algebraic connectivity (λ2) + (optionally) Fiedler/next vectors via SciPy or torch

def lam2_with_vectors(L: np.ndarray, backend: str = "scipy", device: str = "cpu"):
    n = L.shape[0]
    if backend == "scipy":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy not available; use --spectral_backend torch")
        try:
            if n <= 256:
                w, v = np.linalg.eigh(L)
            else:
                w, v = eigsh(csr_matrix(L), k=min(3, n-1), which='SM', tol=1e-6, maxiter=max(1000, 10*n))
        except Exception:
            w, v = np.linalg.eigh(L)

        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(v)):
            w, v = np.linalg.eigh(L)

        idx = np.argsort(w)
        w, v = w[idx], v[:, idx]
        lam2 = float(w[1]) if len(w) > 1 else 0.0
        phi2 = v[:, 1] if v.shape[1] > 1 else np.zeros(n)
        phi3 = v[:, 2] if v.shape[1] > 2 else np.zeros(n)
        return lam2, phi2, phi3
    else:
        tL = torch.as_tensor(L, dtype=torch.float64, device=device)
        w, v = torch.linalg.eigh(tL)
        w = w.cpu().numpy(); v = v.cpu().numpy()
        idx = np.argsort(w)
        w, v = w[idx], v[:, idx]
        lam2 = float(w[1]) if len(w) > 1 else 0.0
        phi2 = v[:, 1] if v.shape[1] > 1 else np.zeros(L.shape[0])
        phi3 = v[:, 2] if v.shape[1] > 2 else np.zeros(L.shape[0])
        return lam2, phi2, phi3


def er_topk(n:int, adj: np.ndarray, k: int, frac_cap: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    L = laplacian_from_adj(adj)
    w, V = np.linalg.eigh(L)
    tol = 1e-12
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    Lplus = (V * invw) @ V.T  

    iu, iv = np.triu_indices(n, k=1)
    mask_non = (adj[iu, iv] == 0)
    iu, iv = iu[mask_non], iv[mask_non]
    er = Lplus[iu, iu] + Lplus[iv, iv] - 2.0 * Lplus[iu, iv]

    total = len(er)
    if frac_cap > 0:
        cap = int(math.ceil(frac_cap * total))
        k_eff = min(k, cap)
    else:
        k_eff = k
    kk = min(k_eff, total)
    if kk <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    idx = np.argpartition(-er, kk-1)[:kk]
    order = np.argsort(-er[idx])
    idx = idx[order]
    return iu[idx], iv[idx], er[idx]


def effective_resistance_for_pairs(adj: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute effective resistance R_eff(u,v) for a batch of pairs on the current graph.
    pairs: (K,2) int ndarray of (u,v) with u<v, no self-edges.
    Uses Laplacian pseudoinverse via full eigen decomposition (n<=~64 typical here).
    """
    n = adj.shape[0]
    L = laplacian_from_adj(adj)
    # Full eigen for stability on small n
    w, V = np.linalg.eigh(L)
    tol = 1e-12
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    Lplus = (V * invw) @ V.T
    u = pairs[:, 0]
    v = pairs[:, 1]
    er = Lplus[u, u] + Lplus[v, v] - 2.0 * Lplus[u, v]
    return er.astype(np.float64)


# -----------------------------
# Graphicality check (Erdős–Gallai)
# -----------------------------
def _is_graphical_erdos_gallai(seq_in: np.ndarray) -> bool:
    """Return True if degree sequence 'seq_in' (non-negative ints) is graphical.
    Uses the Erdős–Gallai theorem. Empty or all-zero sequences are graphical.
    """
    if seq_in is None:
        return True
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
    s.sort()
    s = s[::-1]  # non-increasing
    prefix = np.cumsum(s)
    # precompute tail mins for efficiency
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


# -----------------------------
# Greedy ER baseline (λ2 after greedy ER completion)
# -----------------------------
def er_greedy_baseline_lambda2(n: int, adj: np.ndarray, m: int, backend: str = 'scipy', device: str = 'cpu') -> float:
    """Greedy ER completion to m edges. Uses SMW fast path once connected.
    Computes λ2 at the end only (eigen once).
    """
    adj_b = adj.copy()
    base_edges = int(adj_b.sum() // 2)
    steps = max(0, m - base_edges)
    if steps <= 0:
        Lb = laplacian_from_adj(adj_b)
        lam2_b, _, _ = lam2_with_vectors(Lb, backend=backend, device=device)
        return lam2_b
    use_green = False
    G = None; dG = None
    if is_connected_adj(adj_b):
        try:
            G, dG = G_from_adj(adj_b)
            use_green = True
        except Exception:
            use_green = False
    for _ in range(steps):
        if use_green and (G is not None):
            iu, iv, er = er_topk_fast_from_G(adj_b, G, dG, k=1, frac_cap=1.0)
        else:
            iu, iv, er = er_topk(n, adj_b, k=1, frac_cap=1.0)
        if len(iu) == 0:
            break
        u = int(iu[0]); v = int(iv[0])
        if adj_b[u, v] == 0.0:
            adj_b[u, v] = 1.0
            adj_b[v, u] = 1.0
            if use_green and (G is not None):
                try:
                    G, dG = G_rank1_update_edge(G, dG, u, v, 1.0)
                except Exception:
                    use_green = False
            else:
                if not use_green and is_connected_adj(adj_b):
                    try:
                        G, dG = G_from_adj(adj_b)
                        use_green = True
                    except Exception:
                        use_green = False
    Lb = laplacian_from_adj(adj_b)
    lam2_b, _, _ = lam2_with_vectors(Lb, backend=backend, device=device)
    return lam2_b

def lam2_after_add_and_fill(n: int, adj: np.ndarray, m: int, u: int, v: int, backend: str = 'scipy', device: str = 'cpu') -> float:
    """Terminal λ2 after adding (u,v) then greedy-completing to m by ER.
    Fast path: SMW updates for G once connected; eigen only at the end.
    """
    adj_b = adj.copy()
    if u != v and adj_b[u, v] == 0.0:
        adj_b[u, v] = 1.0
        adj_b[v, u] = 1.0
    return er_greedy_baseline_lambda2(n, adj_b, m, backend=backend, device=device)

# -----------------------------
# Environment with fast spectral cache
# -----------------------------

@dataclass
class EnvConfig:
    n: int
    m: int
    init: str = 'path'  # 'path' or 'empty'
    spectral_backend: str = 'scipy'
    device: str = 'cpu'
    fast_spectral: bool = True
    spectral_refresh_k: int = 1

class GraphBuildEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.n = cfg.n; self.m = cfg.m
        self.backend = cfg.spectral_backend
        self.device = cfg.device
        self.fast = cfg.fast_spectral
        self.refresh_k = max(1, int(cfg.spectral_refresh_k))
        self.reset()

    def _init_adj(self):
        if self.cfg.init == 'path':
            adj = np.zeros((self.n, self.n), dtype=np.float64)
            for i in range(self.n-1):
                adj[i, i+1] = 1.0; adj[i+1, i] = 1.0
            return adj
        else:
            return np.zeros((self.n, self.n), dtype=np.float64)

    def reset(self):
        self.adj = self._init_adj()
        self.base_edges = int(self.adj.sum()/2)
        self.tgt_edges = self.m
        self.max_steps = max(0, self.tgt_edges - self.base_edges)
        self.step_count = 0
        self.done = (self.max_steps == 0)
        # Initial spectral cache
        L = laplacian_from_adj(self.adj)
        lam2, phi2, phi3 = lam2_with_vectors(L, backend=self.backend, device=self.device)
        self.cached = {
            'lam2': lam2,
            'phi2': phi2.astype(np.float32),
            'phi3': phi3.astype(np.float32),
            'deg': self.adj.sum(axis=1).astype(np.float32),
        }
        # Green's function (Laplacian pseudoinverse) cache for fast ER updates
        self.use_green = False
        self.G = None
        self.diagG = None
        if is_connected_adj(self.adj):
            try:
                G, dG = G_from_adj(self.adj)
                self.G, self.diagG = G, dG
                self.use_green = True
            except Exception:
                self.use_green = False
        return self._state(refresh=False)

    def _state(self, refresh: bool) -> dict:
        if refresh:
            L = laplacian_from_adj(self.adj)
            lam2, phi2, phi3 = lam2_with_vectors(L, backend=self.backend, device=self.device)
            self.cached['lam2'] = lam2
            self.cached['phi2'] = phi2.astype(np.float32)
            self.cached['phi3'] = phi3.astype(np.float32)
            self.cached['deg'] = self.adj.sum(axis=1).astype(np.float32)

        n = self.n
        progress = 0.0 if self.max_steps==0 else (self.step_count / self.max_steps)
        dens_target = normalized_density(n, self.m)

        # --- normalize like v8 and build complex chart z ---
        phi2 = self.cached['phi2'].astype(np.float64)
        phi3 = self.cached['phi3'].astype(np.float64)
        phi2 = phi2 - phi2.mean();  phi2 = phi2 / (np.linalg.norm(phi2) + 1e-12)
        phi3 = phi3 - phi3.mean();  phi3 = phi3 / (np.linalg.norm(phi3) + 1e-12)
        z = (phi2 + 1j*phi3).astype(np.complex128)

        # degree normalization like v8
        deg = self.cached['deg'].astype(np.float32)
        deg_norm = deg / max(1.0, float(deg.max()))

        # GAT node features: [deg_norm, Re(z), Im(z), phi2, phi3]
        node_feats = np.stack([
            deg_norm.astype(np.float32),
            np.real(z).astype(np.float32),
            np.imag(z).astype(np.float32),
            phi2.astype(np.float32),
            phi3.astype(np.float32),
        ], axis=1)

        return {
            'node_feats': node_feats,                     # np float32, shape (n,5)
            'lam2': float(self.cached['lam2']),
            'progress': float(progress),
            'dens_target': float(dens_target),
            'z': z,                                       # np complex128, shape (n,)
            'deg_norm': deg_norm.astype(np.float32),      # np float32, shape (n,)
        }

    def step(self, action: Tuple[int,int], refresh_now: bool=False):
        if self.done:
            raise RuntimeError('step() called on done env')
        u, v = action
        if u != v and self.adj[u, v] == 0.0:
            self.adj[u, v] = 1.0; self.adj[v, u] = 1.0
            # Update Green's function if available; else check if we can initialize it now
            if self.use_green and (self.G is not None):
                try:
                    self.G, self.diagG = G_rank1_update_edge(self.G, self.diagG, u, v, 1.0)
                except Exception:
                    self.use_green = False
            else:
                # If just became connected, initialize G once
                if is_connected_adj(self.adj):
                    try:
                        G, dG = G_from_adj(self.adj)
                        self.G, self.diagG = G, dG
                        self.use_green = True
                    except Exception:
                        self.use_green = False
        self.step_count += 1
        self.done = (self.step_count >= self.max_steps)
        # Fast mode: no per-step λ2 (reward=0) and defer spectral refresh
        need_refresh = (not self.fast) or refresh_now or self.done or (self.step_count % self.refresh_k == 0)
        state = self._state(refresh=need_refresh)
        reward = 0.0  # sparse reward only
        info = {}
        return state, reward, self.done, info

    def final_eval(self) -> float:
        L = laplacian_from_adj(self.adj)
        lam2, _, _ = lam2_with_vectors(L, backend=self.backend, device=self.device)
        return lam2

# -----------------------------
# GAT encoder (compact custom)
# -----------------------------

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, bias=True):
        super().__init__()
        self.heads = heads
        self.W = nn.Parameter(torch.randn(heads, in_dim, out_dim) * (1.0/math.sqrt(in_dim)))
        self.a = nn.Parameter(torch.randn(heads, out_dim*2) * 0.01)
        self.leaky = nn.LeakyReLU(0.2)
        self.bias = nn.Parameter(torch.zeros(heads, out_dim)) if bias else None

    def forward(self, x, adj):
        # x: (n, in_dim), adj: (n,n) 0/1 (include self-loops)
        n = x.size(0)
        h = torch.einsum('hid,ni->hnd', self.W, x)  # (h, n, out)
        # compute attention per head
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        cat = torch.cat([h_i, h_j], dim=-1)  # (h,n,n,2*out)
        e = self.leaky(torch.einsum('hD,hndD->hnd', self.a, cat))
        # mask with adjacency (add self-loops)
        attn_mask = (adj > 0).unsqueeze(0)
        e = e.masked_fill(~attn_mask, -1e9)

        # normalize over *sources i* for each target j (incoming attention), as in v8
        a = torch.softmax(e, dim=1)  # (H, n_i, n_j)

        # aggregate to targets: h'_j = Σ_i α_{ij} * h_i
        out = torch.einsum('hij,hid->hjd', a, h)  # (H, n, out_dim)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)     # (H, n, out_dim)
        out = out.permute(1, 0, 2).reshape(n, -1)  # (n, H*out_dim)
        return out

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(GATLayer(d, hid, heads=heads))
            d = hid*heads
        self.out_dim = d

    def forward(self, x, adj):
        # ensure self-loops
        adj = adj.clone()
        diag = torch.eye(adj.size(0), device=adj.device)
        adj = torch.maximum(adj, diag)
        for g in self.layers:
            x = g(x, adj)
            x = F.elu(x)
        return x  # (n, out_dim)

# -----------------------------
# Unified policy/value net (single model, no buckets)
# -----------------------------

class PolicyValueNet(nn.Module):

    def __init__(self, node_in, hid, heads, layers, edge_hidden, value_hidden, device='cpu'):
        super().__init__()
        self.encoder = GATEncoder(node_in, hid, heads, layers)
        self.node_emb_dim = self.encoder.out_dim

        self.gc_query = nn.Parameter(torch.randn(self.node_emb_dim) * 0.01)
        
        # CHANGED: global_feats is now 5-dimensional, so edge_in and value_in increase by 1
        # UPDATED: edge_feats: 7 -> 11 (regularity augmentations)
        edge_in = self.node_emb_dim * 2 + self.node_emb_dim + 11 + 5
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        self.q_mlp = nn.Sequential(
            nn.Linear(edge_in, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        # CHANGED: value_in increases by 1 for the new global feature
        value_in = self.node_emb_dim + 5
        self.value_mlp = nn.Sequential(
            nn.Linear(value_in, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, node_feats, adj_dense, cand_edges, edge_feats_7, global_feats_5): # CHANGED: Renamed for clarity
        """
        node_feats:     (n, F)
        adj_dense:      (n, n)
        cand_edges:     (K, 2) long
        edge_feats_7:   (K, 7)
        global_feats_5: (5,)   -> [log1p(n), dens_target, progress, lam2, is_regular_possible]
        """
        x = self.encoder(node_feats, adj_dense)

        pool = x.mean(dim=0)
        gfull = torch.cat([pool, global_feats_5], dim=0)
        value = self.value_mlp(gfull).squeeze(-1)

        if cand_edges.numel() == 0:
            return torch.empty(0, device=x.device), value, torch.empty(0, device=x.device)

        scores = torch.matmul(x, self.gc_query)
        alpha = torch.softmax(scores, dim=0).unsqueeze(1)
        gc = (alpha * x).sum(dim=0)
        
        u, v = cand_edges[:, 0], cand_edges[:, 1]
        xu, xv = x[u], x[v]
        uv = torch.cat([xu, xv], dim=1)

        K = cand_edges.size(0)
        gc_rep = gc.unsqueeze(0).expand(K, -1)
        gf_rep = global_feats_5.unsqueeze(0).expand(K, -1) # CHANGED

        edge_full = torch.cat([uv, gc_rep, edge_feats_7, gf_rep], dim=1)
        edge_full = torch.nan_to_num(edge_full, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.edge_mlp(edge_full).squeeze(-1)
        q_pred = self.q_mlp(edge_full).squeeze(-1)
        # Guard against any numerical issues
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        q_pred = torch.nan_to_num(q_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return logits, value, q_pred
    

# -----------------------------
# PPO Agent (final-return reward)
# -----------------------------

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    ppo_epochs: int = 4
    mb_size: int = 4096
    train_temperature: float = 1.0
    target_kl: float = 0.04
    clip_vf: float = 0.2
    lr_min: float = 5e-5
    aux_q_coef: float = 0.5
    # Stabilizers
    ema_decay: float = 0.99
    consistency_coef: float = 0.0  # KL(student||EMA) coefficient
    consistency_decay: float = 1.0 # multiply after each PPO update
    adv_clip: float = 0.0          # clip normalized advantages to [-adv_clip, +adv_clip] (0=off)

class PPOAgent:
    def __init__(self, net: PolicyValueNet, cfg: PPOConfig, device='cpu'):
        self.net = net
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        # Exponential moving average (teacher) network for stable behavior + regularization
        self.ema_net = copy.deepcopy(self.net).to(device)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)
        self.clear_buffer()

    def clear_buffer(self):
        self.obs = []  # tuples holding tensors already on device
        self.acts = []
        self.logps = []
        self.rews = []
        self.vals = []
        self.dones = []

    def store(self, obs_tensors, act_idx, logp, val, rew, done):
        self.obs.append(obs_tensors)
        self.acts.append(act_idx)
        self.logps.append(logp)
        self.rews.append(rew)
        self.vals.append(val)
        self.dones.append(done)

    @torch.no_grad()
    def _ema_update(self, decay: float):
        d = decay
        for p_t, p_s in zip(self.ema_net.parameters(), self.net.parameters()):
            p_t.data.mul_(d).add_(p_s.data, alpha=(1.0 - d))

    def _compute_adv(self, rewards, values, dones, gamma, lam):
        T = len(rewards)
        adv = torch.zeros(T, device=self.device)
        lastgaelam = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def ppo_update(self):
        if not self.obs:
            return {'loss':0.0,'pol':0.0,'val':0.0,'ent':0.0,'kl':0.0,'clipfrac':0.0,'q':0.0}
        cfg = self.cfg
        # Stack rollout
        acts = torch.stack(self.acts).detach()
        old_logp = torch.stack(self.logps).detach()
        vals = torch.stack(self.vals).detach()
        rews = torch.tensor(self.rews, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            adv, ret = self._compute_adv(rews, vals, dones, cfg.gamma, cfg.lam)
            # Robust normalization: avoid NaNs when T<=1 or zero-variance
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False)
            if not torch.isfinite(adv_std) or adv_std.item() == 0.0:
                adv = adv - adv_mean
            else:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            # Final safety
            adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
            ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
            # Optional advantage clipping
            if cfg.adv_clip and cfg.adv_clip > 0.0:
                adv = adv.clamp(-cfg.adv_clip, +cfg.adv_clip)

        # Flatten obs into a list of lazy callables to re-forward per minibatch
        data = list(zip(self.obs, acts, old_logp, vals, adv, ret))
        mbsize = min(cfg.mb_size, len(data))
        losses = {'loss':0.0,'pol':0.0,'val':0.0,'ent':0.0,'kl':0.0,'clipfrac':0.0,'q':0.0}
        for _ in range(cfg.ppo_epochs):
            random.shuffle(data)
            for i in range(0, len(data), mbsize):
                batch = data[i:i+mbsize]
                if not batch: break
                new_logps=[]; entropies=[]; values=[]; oldlog=[]; actidx=[]; advantages=[]; returns=[]
                old_values_list=[]; q_losses=[]; cons_losses=[]
                for (obs_t, a, olp, v, gae, R) in batch:
                    (node_feats, adj_dense, cand_edges, edge_feats, global_feats, temp, *rest) = obs_t
                    # Sanitize inputs to be safe against any accidental NaNs/Infs
                    node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
                    adj_dense  = torch.nan_to_num(adj_dense,  nan=0.0, posinf=0.0, neginf=0.0)
                    edge_feats = torch.nan_to_num(edge_feats, nan=0.0, posinf=0.0, neginf=0.0)
                    global_feats = torch.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0)

                    meta = rest[0] if len(rest) > 0 else None
                    q_aux = None
                    if isinstance(meta, dict):
                        q_aux = meta.get('q_aux', None)
                    logits, value, q_pred = self.net(node_feats, adj_dense, cand_edges, edge_feats, global_feats)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                    dist = Categorical(logits=logits / temp)
                    new_logps.append(dist.log_prob(a))
                    entropies.append(dist.entropy())
                    values.append(value)
                    oldlog.append(olp)
                    actidx.append(a)
                    advantages.append(gae)
                    returns.append(R)
                    old_values_list.append(v)

                    # Consistency regularization to EMA teacher (no grad through teacher)
                    if cfg.consistency_coef and cfg.consistency_coef > 0.0:
                        with torch.no_grad():
                            logits_t, _, _ = self.ema_net(node_feats, adj_dense, cand_edges, edge_feats, global_feats)
                            logits_t = torch.nan_to_num(logits_t, nan=0.0, posinf=1e6, neginf=-1e6)
                        p = torch.softmax(logits / temp, dim=-1)
                        logp = torch.log_softmax(logits / temp, dim=-1)
                        q = torch.softmax(logits_t / temp, dim=-1)
                        # KL(P||Q) across candidates
                        kl_pq = torch.sum(p * (logp - torch.log(q + 1e-8)), dim=-1)
                        cons_losses.append(kl_pq.mean())

                    # Auxiliary Q MSE on available targets
                    if q_aux is not None:
                        # q_aux: (idx_chosen, q_target_chosen, er_idx(=0), q_target_er1)
                        idx_c, qt_c, idx_e, qt_e = q_aux
                        # guard against empty candidate set
                        if q_pred.numel() > 0:
                            pred_c = q_pred[int(idx_c)]
                            pred_e = q_pred[int(idx_e)]
                            q_loss = F.mse_loss(pred_c, torch.as_tensor(qt_c, dtype=torch.float32, device=self.device)) + \
                                     F.mse_loss(pred_e, torch.as_tensor(qt_e, dtype=torch.float32, device=self.device))
                            q_losses.append(q_loss)
                new_logps = torch.stack(new_logps)
                entropies = torch.stack(entropies)
                values = torch.stack(values)
                oldlog = torch.stack(oldlog)
                advantages = torch.stack(advantages)
                returns = torch.stack(returns)
                old_values = torch.stack(old_values_list)
                cons_loss_total = (torch.stack(cons_losses).mean() if len(cons_losses)>0 else torch.tensor(0.0, device=self.device))

                ratio = torch.exp(new_logps - oldlog)
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0)
                clip_adv = torch.clamp(ratio, 1.0-cfg.clip, 1.0+cfg.clip) * advantages
                pol_loss = -torch.min(ratio*advantages, clip_adv).mean()
                if self.cfg.clip_vf and self.cfg.clip_vf > 0.0:
                    values_clipped = old_values + (values - old_values).clamp(-self.cfg.clip_vf, self.cfg.clip_vf)
                    v_loss_unclipped = F.mse_loss(values, returns)
                    v_loss_clipped   = F.mse_loss(values_clipped, returns)
                    val_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    val_loss = 0.5 * F.mse_loss(values, returns)
                ent_loss = -cfg.ent_coef * entropies.mean()
                q_loss_total = torch.stack(q_losses).mean() if len(q_losses)>0 else torch.tensor(0.0, device=self.device)
                loss = pol_loss + cfg.vf_coef*val_loss + ent_loss + cfg.aux_q_coef * q_loss_total + cfg.consistency_coef * cons_loss_total

                # A tiny KL readout
                with torch.no_grad():
                    approx_kl = (oldlog - new_logps).mean().clamp_min(0.0)
                    clipfrac = (torch.abs(ratio-1.0) > cfg.clip).float().mean()

                # Early stop on large update to avoid collapse
                if approx_kl.item() > self.cfg.target_kl * 1.5:
                    # perform a lighter update (entropy-only) and break out
                    self.opt.zero_grad(set_to_none=True)
                    ( -self.cfg.ent_coef * entropies.mean() ).backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.opt.step()
                    # keep EMA in sync softly even on light step
                    self._ema_update(self.cfg.ema_decay)
                    # accumulate logs and break early from this epoch
                    losses['kl'] += float(approx_kl.detach().cpu())
                    losses['clipfrac'] += float(((torch.abs(ratio-1.0) > self.cfg.clip).float().mean()).detach().cpu())
                    # Skip the normal update for this batch
                    continue

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                # Update EMA teacher
                self._ema_update(self.cfg.ema_decay)

                losses['loss'] += float(loss.detach().cpu())
                losses['pol']  += float(pol_loss.detach().cpu())
                losses['val']  += float(val_loss.detach().cpu())
                losses['ent']  += float((-ent_loss/cfg.ent_coef).detach().cpu())
                losses['kl']   += float(approx_kl.detach().cpu())
                losses['clipfrac'] += float(clipfrac.detach().cpu())
                losses['q']  += float(q_loss_total.detach().cpu())

        # average by number of minibatches processed
        denom = max(1, (len(data)+mbsize-1)//mbsize * cfg.ppo_epochs)
        for k in losses: losses[k] /= denom
        # Decay consistency regularizer if requested
        if self.cfg.consistency_decay and self.cfg.consistency_decay != 1.0:
            self.cfg.consistency_coef = float(self.cfg.consistency_coef) * float(self.cfg.consistency_decay)
        self.clear_buffer()
        return losses

    # Save/load with arch metadata
    def save(self, path:str, arch:dict, prefer_ema: bool = False):
        state = self.ema_net.state_dict() if prefer_ema else self.net.state_dict()
        obj = {
            'arch': arch,
            'state_dict': state,
            'opt': self.opt.state_dict(),
            'ema_state_dict': self.ema_net.state_dict(),
        }
        torch.save(obj, path)

    def load(self, path:str, device):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message=r"You are using `torch\.load`.*weights_only=False",category=FutureWarning)
            obj = torch.load(path, map_location=device)  # keep compatibility with older torch
        return obj

# -----------------------------
# Training / Inference loop
# -----------------------------

# Normalized density in [0,1]: 0 at base tree (n-1 edges), 1 at complete graph
def normalized_density(n: int, m: int) -> float:
    Mmax = n*(n-1)/2.0
    base = float(n-1)
    denom = max(1.0, Mmax - base)
    d = (float(m) - base) / denom
    return float(min(1.0, max(0.0, d)))

def bucket_id_from_density(n:int, m:int, buckets:int)->int:
    dens = normalized_density(n, m)
    bid = int(min(buckets-1, max(0, math.floor(dens * buckets))))
    return bid

def size_bucket_id(n:int, n_min:int, n_max:int, buckets:int)->int:
    if buckets <= 1:
        return 0
    span = max(1, n_max - n_min + 1)
    # Equal-width integer bins
    width = math.ceil(span / buckets)
    idx = (n - n_min) // max(1, width)
    return int(min(buckets-1, max(0, idx)))

def composite_bucket_id(n:int, m:int, n_min:int, n_max:int, n_buckets:int, dens_buckets:int)->int:
    bi = size_bucket_id(n, n_min, n_max, n_buckets)
    bj = bucket_id_from_density(n, m, dens_buckets)
    return int(bi * dens_buckets + bj)

def run_episode(env: GraphBuildEnv, net: PolicyValueNet, agent: PPOAgent, args, train=True, ep: int = 0):
    # Greedy ER baseline for terminal reward (for logging/optionally final assignment)
    adj0 = env.adj.copy()
    base_lam2 = er_greedy_baseline_lambda2(env.n, adj0, env.m, backend=env.backend, device=env.device)

    temp = torch.tensor([args.train_temperature if train else 1.0], device=agent.device).squeeze(0)

    done = env.done
    total_steps = 0
    # Diagnostics accumulators ...
    diag_acc = {
        'top1_hits': 0,
        'top5_hits': 0,
        'rank_sum': 0.0,
        'entropy_sum': 0.0,
        'corr_sum': 0.0,
        'corr_count': 0
    }
    # (keep the rest of run_episode unchanged)

    # Stepwise ratio shaping hyperparams (minimal knobs)
    ro_eps   = float(getattr(args, 'rollout_reward_eps', 1e-6))
    ro_scale = float(getattr(args, 'rollout_reward_scale', 1.0))
    ro_clip  = float(getattr(args, 'rollout_reward_clip', 0.0))

    while not done:
        # Current cached state (no refresh)
        st = env._state(refresh=False)
        
        # Determine two-phase regular-first targets
        n = env.n; m = env.m
        # Use live degrees for accuracy (cache may be stale without refresh)
        deg_abs = env.adj.sum(axis=1).astype(np.float32)
        e_now = int(env.adj.sum() // 2)
        max_deg = float(deg_abs.max()) if deg_abs.size > 0 else 0.0
        reg_possible = ((2 * m) % n == 0)
        if reg_possible:
            k_reg_target = int((2 * m) // n)  # exact k
            m1 = m
        else:
            k_reg_target = int(math.floor((2 * m) / n))
            m1 = (n * k_reg_target) // 2
        # Phase-1 if we can still add edges and there exist deficits relative to k_tgt
        deficits = np.maximum(0.0, k_reg_target - deg_abs)
        have_deficit = bool(np.any(deficits > 0.5))  # tolerate float
        phase1 = (e_now < m1) and have_deficit and (k_reg_target > 0) and (k_reg_target >= int(math.floor(max_deg)))

        # Candidate generation
        # Always compute ER Top-K as a robust baseline (fast if connected & G available)
        if getattr(env, 'use_green', False) and (env.G is not None) and (env.diagG is not None):
            iu_er, iv_er, ers_er = er_topk_fast_from_G(env.adj, env.G, env.diagG, k=args.topk, frac_cap=args.topk_frac)
        else:
            iu_er, iv_er, ers_er = er_topk(n, env.adj, k=args.topk, frac_cap=args.topk_frac)
        if len(iu_er) == 0:
            _ = env._state(refresh=True)
            break

        # Build deficit-driven candidates in Phase-1 to steer toward near-regular
        if phase1:
            # All non-edges between deficit nodes
            uu_all, vv_all = np.triu_indices(n, k=1)
            mask_non = (env.adj[uu_all, vv_all] == 0)
            uu_all = uu_all[mask_non]; vv_all = vv_all[mask_non]
            # filter to deficit pairs
            if uu_all.size > 0:
                def_u_all = (k_reg_target - deg_abs[uu_all]).astype(np.float32)
                def_v_all = (k_reg_target - deg_abs[vv_all]).astype(np.float32)
                mask_def = (def_u_all > 0.5) & (def_v_all > 0.5)
                uu_all = uu_all[mask_def]; vv_all = vv_all[mask_def]

            # Erdos-Gallai feasibility filter per candidate (on residual deficits, ignoring current adjacency constraints)
            cand_u = []
            cand_v = []
            if uu_all.size > 0:
                deficits_int = np.maximum(0, np.rint(k_reg_target - deg_abs).astype(np.int64))
                for uu_i, vv_i in zip(uu_all, vv_all):
                    if deficits_int[uu_i] <= 0 or deficits_int[vv_i] <= 0:
                        continue
                    seq = deficits_int.copy()
                    seq[uu_i] -= 1
                    seq[vv_i] -= 1
                    if _is_graphical_erdos_gallai(seq):
                        cand_u.append(int(uu_i)); cand_v.append(int(vv_i))

            if len(cand_u) == 0:
                # Havel–Hakimi style fallback: connect highest deficits greedily if possible
                deficits_int = np.maximum(0, np.rint(k_reg_target - deg_abs).astype(np.int64))
                nodes = np.argsort(-deficits_int)  # descending deficits
                picked = False
                for i_idx in range(len(nodes)):
                    u0 = int(nodes[i_idx])
                    if deficits_int[u0] <= 0:
                        break
                    # pick a v among next highest deficits, non-adjacent to u0
                    for j_idx in range(i_idx + 1, len(nodes)):
                        v0 = int(nodes[j_idx])
                        if deficits_int[v0] <= 0:
                            continue
                        if env.adj[u0, v0] == 0.0 and u0 != v0:
                            cand_u = [u0]; cand_v = [v0]
                            picked = True
                            break
                    if picked:
                        break

            if len(cand_u) > 0:
                cu = np.array(cand_u, dtype=np.int64)
                cv = np.array(cand_v, dtype=np.int64)
                # Compute ER for deficit candidates (fast via Green if available)
                pairs = np.stack([cu, cv], axis=1)
                try:
                    if getattr(env, 'use_green', False) and (env.G is not None) and (env.diagG is not None):
                        ers_def = er_pairs_from_G(env.G, env.diagG, pairs).astype(np.float32)
                    else:
                        ers_def = effective_resistance_for_pairs(env.adj, pairs).astype(np.float32)
                except Exception:
                    ers_def = np.zeros((pairs.shape[0],), dtype=np.float32)

                # Compute Fiedler gain proxy (phi2 difference squared)
                phi2_vec = st['node_feats'][:, 3].astype(np.float32)
                phi2_gain = (phi2_vec[cu] - phi2_vec[cv]) ** 2

                # Degree deficits for regularization potential reduction
                def_u = (k_reg_target - deg_abs[cu]).astype(np.float32)
                def_v = (k_reg_target - deg_abs[cv]).astype(np.float32)

                # Adaptive weights: emphasize regularity early, spectral later
                total_def = float(np.maximum(0.0, k_reg_target - deg_abs).sum())
                denom_def = float(max(1.0, n * max(1, k_reg_target)))
                w_reg = min(1.0, max(0.0, total_def / denom_def))
                alpha = w_reg
                beta  = (1.0 - w_reg) * 0.30
                gamma = (1.0 - w_reg) * 0.70

                score = alpha * (def_u + def_v) + beta * ers_def + gamma * phi2_gain

                # Preselect top-K deficit candidates by the composite score
                K_def = int(max(1, args.topk))
                if score.size > K_def:
                    idx = np.argpartition(-score, K_def-1)[:K_def]
                    order = np.argsort(-score[idx])
                    idx = idx[order]
                    cu, cv = cu[idx], cv[idx]
                    ers_def = ers_def[idx]
                iu = cu; iv = cv; ers = ers_def
            else:
                # Final fallback to ER Top-K if no feasible deficit pair exists
                iu = iu_er; iv = iv_er; ers = ers_er
        else:
            # Phase-2 (refinement): use ER Top-K directly (optionally could blend degree-variance pairs)
            iu = iu_er; iv = iv_er; ers = ers_er

        # Optional candidate augmentation with random non-ER edges during Phase-2 only (exploration)
        if (not phase1) and getattr(args, 'rand_cand_frac', 0.0) and args.rand_cand_frac > 0.0:
            K = len(iu)
            if K > 0:
                uu_all, vv_all = np.triu_indices(n, k=1)
                mask_non = (env.adj[uu_all, vv_all] == 0)
                uu_all = uu_all[mask_non]; vv_all = vv_all[mask_non]
                cand_set = set(zip(iu.tolist(), iv.tolist()))
                pool = [(int(uu_all[t]), int(vv_all[t])) for t in range(len(uu_all)) if (int(uu_all[t]), int(vv_all[t])) not in cand_set]
                if len(pool) > 0:
                    K_rand = min(min(int(math.ceil(args.rand_cand_frac * K)), int(getattr(args, 'rand_cand_max', K))), len(pool))
                    if K_rand > 0:
                        rand_pairs = random.sample(pool, K_rand)
                        rp = np.array(rand_pairs, dtype=np.int64)
                        try:
                            if getattr(env, 'use_green', False) and (env.G is not None) and (env.diagG is not None):
                                ers_rand = er_pairs_from_G(env.G, env.diagG, rp).astype(np.float32)
                            else:
                                ers_rand = effective_resistance_for_pairs(env.adj, rp).astype(np.float32)
                        except Exception:
                            ers_rand = np.zeros((K_rand,), dtype=np.float32)
                        iu = np.concatenate([iu, rp[:,0]])
                        iv = np.concatenate([iv, rp[:,1]])
                        ers = np.concatenate([ers, ers_rand])
        if len(iu) == 0:
            _ = env._state(refresh=True)
            break
        node_feats = torch.tensor(st['node_feats'], dtype=torch.float32, device=agent.device)
        adj_dense = torch.tensor(env.adj, dtype=torch.float32, device=agent.device)
        cand_edges = torch.tensor(np.stack([iu,iv], axis=1), dtype=torch.long, device=agent.device)

        # v8-style edge feats + regularity-aware augmentations
        z = st['z']                      # complex (n,)
        degn = st['deg_norm']            # (n,) degree normalized to [0,1]
        progress = float(st['progress'])
        dens_target = float(st['dens_target'])
        lam2_now = float(st['lam2'])
        is_regular_possible = 1.0 if (2 * env.m) % env.n == 0 else 0.0

        # Candidate endpoints
        u = iu; v = iv
        K = len(u)

        # Original cues
        dz = np.abs(z[u] - z[v]).astype(np.float32)
        du_norm = degn[u].astype(np.float32)
        dv_norm = degn[v].astype(np.float32)
        hs = ers.astype(np.float32)

        # Absolute degrees (for regularity math)
        deg_abs = env.cached['deg'].astype(np.float32)
        du_abs = deg_abs[u]
        dv_abs = deg_abs[v]

        # Current edge count, mean degree, and variance
        e_now = int(env.adj.sum() // 2)
        mu = 2.0 * e_now / env.n
        S2 = float((deg_abs**2).sum())
        Var = S2 / env.n - mu * mu

        # Target regular degree if realizable
        k_tgt = math.ceil(2.0 * env.m / env.n)

        # End-point degree deficits toward k_tgt (floor at 0)
        def_u = np.maximum(0.0, k_tgt - du_abs).astype(np.float32)
        def_v = np.maximum(0.0, k_tgt - dv_abs).astype(np.float32)

        # Exact change in degree variance if we add (u,v)
        # ΔVar = (2*(deg[u]+deg[v])+2)/n - (4*μ/n + 4/n^2)
        dVar = (2.0 * (du_abs + dv_abs) + 2.0) / env.n - (4.0 * mu / env.n + 4.0 / (env.n * env.n))
        dVar = dVar.astype(np.float32)

        # Two-hop overlap |N(u) ∩ N(v)| normalized by (n-2)
        A = env.adj  # (n,n) 0/1
        Au = A[u]    # (K,n)
        Av = A[v]    # (K,n)
        overlap = (Au * Av).sum(axis=1).astype(np.float32)
        den = max(1, env.n - 2)
        overlap = overlap / den

        # Stack original 7 + 4 new = 11 edge features
        edge_feats_np = np.stack([
            dz,
            du_norm, dv_norm,
            np.full(K, progress, dtype=np.float32),
            np.full(K, dens_target, dtype=np.float32),
            np.full(K, lam2_now, dtype=np.float32),
            hs,
            def_u, def_v,
            dVar, overlap
        ], axis=1).astype(np.float32)

        edge_feats = torch.tensor(edge_feats_np, dtype=torch.float32, device=agent.device)

        global_feats = torch.tensor([math.log1p(env.n), dens_target, progress, lam2_now, is_regular_possible], dtype=torch.float32, device=agent.device)

        # Optionally act with EMA teacher for added stability
        act_model = net
        if getattr(agent, 'ema_net', None) is not None:
            use_ema_train = train and getattr(args, 'ema_policy', False) and (ep >= int(getattr(args, 'ema_start', 0)))
            use_ema_eval  = (not train) and getattr(args, 'ema_infer', False)
            if use_ema_train or use_ema_eval:
                act_model = agent.ema_net

        logits, value, q_pred = act_model(node_feats, adj_dense, cand_edges, edge_feats, global_feats)
        dist = Categorical(logits=logits / (args.train_temperature if train else 1.0))
        a_idx = dist.sample() if train else torch.argmax(dist.logits)

        # If we are close to completion in Phase-2 during inference, do a cheap lookahead:
        # evaluate λ2 after add+greedy-complete for each candidate and pick the best.
        # This guards against unlucky choices when only a few edges remain.
        if (not train):
            edges_left_total = m - e_now
            if (not phase1) and edges_left_total <= 3:
                try:
                    with torch.no_grad():
                        best_idx = int(a_idx.detach().cpu().item())
                        best_val = -1e30
                        Kc = cand_edges.size(0)
                        for ci in range(Kc):
                            uu_i = int(cand_edges[ci,0].item())
                            vv_i = int(cand_edges[ci,1].item())
                            lam_i = lam2_after_add_and_fill(n, env.adj, m, uu_i, vv_i, backend=env.backend, device=env.device)
                            if lam_i > best_val:
                                best_val = lam_i
                                best_idx = ci
                        a_idx = torch.as_tensor(best_idx, dtype=torch.long, device=agent.device)
                except Exception:
                    pass

        # --- per-step diagnostics ---
        with torch.no_grad():
            ent_step = float(dist.entropy().detach().cpu().item())
            diag_acc['entropy_sum'] += ent_step

            logits_np = logits.detach().cpu().numpy()
            # Pearson proxy for monotonicity vs ER (safe, no warnings)
            if logits_np.size > 1:
                x = logits_np.astype(np.float64, copy=False)
                y = np.asarray(ers, dtype=np.float64)
                x_mean = x.mean(); y_mean = y.mean()
                x_cent = x - x_mean; y_cent = y - y_mean
                sx = float(np.sqrt(np.sum(x_cent * x_cent) / max(1, x_cent.size - 1)))
                sy = float(np.sqrt(np.sum(y_cent * y_cent) / max(1, y_cent.size - 1)))
                if sx > 1e-12 and sy > 1e-12:
                    c = float(np.mean(x_cent * y_cent) / (sx * sy))
                    if np.isfinite(c):
                        diag_acc['corr_sum'] += c
                        diag_acc['corr_count'] += 1

            chosen_k = int(a_idx.detach().cpu().item())
            diag_acc['top1_hits'] += int(chosen_k == 0)
            diag_acc['top5_hits'] += int(chosen_k < 5)

            # ER rank (1 = best ER)
            er_order = np.argsort(-ers)  # descending ER
            er_rank = int(np.where(er_order == chosen_k)[0][0]) + 1
            diag_acc['rank_sum'] += er_rank

        with torch.no_grad():
            old_logp = dist.log_prob(a_idx)
            val0 = value

        (u_pick, v_pick) = (int(cand_edges[a_idx,0].item()), int(cand_edges[a_idx,1].item()))

        # --- Stepwise ratio shaping vs ER-top1 (takes precedence over final-only reward) ---
        r_used = 0.0
        q_aux = None
        if train and getattr(args, 'rollout_reward', False):
            # ER-top1 candidate from this state (er_topk returns sorted desc)
            top1_u = int(cand_edges[0, 0].item())
            top1_v = int(cand_edges[0, 1].item())
            lamA = lam2_after_add_and_fill(env.n, env.adj, env.m, u_pick, v_pick,
                                           backend=env.backend, device=env.device)
            lamB = lam2_after_add_and_fill(env.n, env.adj, env.m, top1_u, top1_v,
                                           backend=env.backend, device=env.device)

            # ratio shaping: (A - B) / max(eps, |B|)
            denom = max(ro_eps, abs(lamB))
            r_step = (lamA - lamB) / denom

            # Optional scaling & clipping for stability (defaults keep identity)
            r_step *= ro_scale
            if ro_clip and ro_clip > 0.0:
                r_step = float(np.clip(r_step, -ro_clip, +ro_clip))

            # Regularity shaping (very small, only when regular k is feasible)
            if is_regular_possible >= 0.5 and Var > 1e-9:
                # Use the dVar of the chosen action: reward reducing variance
                dvar_chosen = float(dVar[int(a_idx.detach().cpu().item())])
                r_reg = (-dvar_chosen) / Var
                r_reg = float(np.clip(r_reg, -0.5, 0.5))  # guardrails
                r_step += args.reg_shaping_coef * r_reg

            r_used = float(r_step)

            # Auxiliary Q targets: terminal λ2 after add+greedy-complete
            q_target_chosen = float(lamA)
            q_target_er1    = float(lamB)
            q_aux = (int(a_idx.detach().cpu().item()), q_target_chosen,
                     0, q_target_er1)  # (chosen_idx, chosen_q), (er1_idx=0, er1_q)

        # --- Detailed per-step logging (JSONL) ---
        if train and getattr(args, 'train_log', None) and args.train_log and (ep % max(1, args.log_detail_every) == 0) and (total_steps <= args.log_max_steps):
            with torch.no_grad():
                logits_cpu = logits.detach().cpu()
                ent = float(dist.entropy().detach().cpu().item())
                top_show = int(min(3, logits_cpu.numel()))
                top_idx = torch.topk(logits_cpu, top_show).indices.tolist()
                chosen_k = int(a_idx.detach().cpu().item())
                # candidate arrays on CPU/np
                u_np = np.asarray(u, dtype=np.int64)
                v_np = np.asarray(v, dtype=np.int64)
                ers_np = np.asarray(ers, dtype=np.float32)
                dz_np = np.asarray(dz, dtype=np.float32)
                du_np = np.asarray(du_norm, dtype=np.float32)
                dv_np = np.asarray(dv_norm, dtype=np.float32)

                rec = {
                    'ep': int(ep),
                    'step': int(total_steps+1),
                    'n': int(env.n),
                    'm': int(env.m),
                    'dens': float(normalized_density(env.n, env.m)),
                    'progress': float(progress),
                    'lambda2_now': float(lam2_now),
                    'K': int(len(u_np)),
                    'train_temperature': float(args.train_temperature),
                    'logits_mean': float(logits_cpu.mean().item()),
                    'logits_std': float(logits_cpu.std(unbiased=False).item()),
                    'logits_max': float(logits_cpu.max().item()),
                    'logits_min': float(logits_cpu.min().item()),
                    'entropy': ent,
                    'value': float(val0.detach().cpu().item()),
                    'chosen_idx': chosen_k,
                    'chosen_u': int(u_np[chosen_k]) if len(u_np)>0 else -1,
                    'chosen_v': int(v_np[chosen_k]) if len(v_np)>0 else -1,
                    'chosen_er': float(ers_np[chosen_k]) if len(ers_np)>0 else 0.0,
                    'chosen_dz': float(dz_np[chosen_k]) if len(dz_np)>0 else 0.0,
                    'chosen_du': float(du_np[chosen_k]) if len(du_np)>0 else 0.0,
                    'chosen_dv': float(dv_np[chosen_k]) if len(dv_np)>0 else 0.0,
                    'er_top1_u': int(u_np[0]) if len(u_np)>0 else -1,
                    'er_top1_v': int(v_np[0]) if len(v_np)>0 else -1,
                    'ers_max': float(ers_np.max()) if len(ers_np)>0 else 0.0,
                    'ers_mean': float(ers_np.mean()) if len(ers_np)>0 else 0.0,
                    'dz_mean': float(dz_np.mean()) if len(dz_np)>0 else 0.0,
                    'dz_std': float(dz_np.std()) if len(dz_np)>0 else 0.0,
                }
                if getattr(args, 'rollout_reward', False):
                    rec.update({
                        'rollout_mode': 'ratio',
                        'rollout_reward': float(r_used),
                        'rollout_scale': float(ro_scale),
                        'lamA_chosen': float(q_target_chosen),
                        'lamB_er_top1': float(q_target_er1),
                    })
                    with torch.no_grad():
                        qpred_cpu = q_pred.detach().cpu().numpy()
                        rec.update({
                            'qpred_chosen': float(qpred_cpu[chosen_k]) if len(qpred_cpu)>0 else 0.0,
                            'qpred_er_top1': float(qpred_cpu[0]) if len(qpred_cpu)>0 else 0.0,
                            'qtarget_chosen': float(q_target_chosen),
                            'qtarget_er_top1': float(q_target_er1),
                            'deviated_from_er': bool(chosen_k != 0),
                        })
                # top-3 by policy logits snapshot
                top_list = []
                for j in top_idx:
                    top_list.append({
                        'idx': int(j),
                        'u': int(u_np[j]),
                        'v': int(v_np[j]),
                        'logit': float(logits_cpu[j].item()),
                        'er': float(ers_np[j]),
                        'dz': float(dz_np[j]),
                    })
                rec['top_logits'] = top_list
                log_jsonl(args.train_log, rec)

        # Apply the chosen action to the true env
        next_state, _, done, info = env.step((u_pick, v_pick), refresh_now=False)

        if train:
            meta = {
                'q_aux': q_aux if getattr(args, 'rollout_reward', False) else None,
            }
            agent.store(
                (
                    node_feats.detach(),
                    adj_dense.detach(),
                    cand_edges.detach(),
                    edge_feats.detach(),          
                    global_feats.detach(),
                    torch.tensor(args.train_temperature, device=agent.device),
                    meta,
                ),
                a_idx.detach(),
                old_logp.detach(),
                val0.detach(),
                float(r_used),
                done
            )
        total_steps += 1

    lam2_final = env.final_eval()

    # Compute reporting metric vs ER baseline
    if args.reward_mode == 'margin':
        reward_log = lam2_final - base_lam2
    elif args.reward_mode == 'ratio':
        denom = max(args.reward_eps, abs(base_lam2))
        reward_log = (lam2_final - base_lam2) / denom
    elif args.reward_mode == 'logratio':
        reward_log = math.log(max(args.reward_eps, lam2_final)) - math.log(max(args.reward_eps, base_lam2))
    else:
        reward_log = lam2_final - base_lam2

    # Optional clip for terminal-only logging/assignment
    if hasattr(args, 'reward_clip') and args.reward_clip and args.reward_clip > 0.0:
        reward_log = float(np.clip(reward_log, -args.reward_clip, args.reward_clip))

    # Only assign terminal reward to the last step if explicitly requested AND not using rollout shaping
    if train and len(agent.rews) > 0 and getattr(args, 'final_return_reward', False) and not getattr(args, 'rollout_reward', False):
        agent.rews[-1] = float(reward_log)

    # --- compute compact diagnostics dict ---
    steps_safe = max(1, total_steps)
    diag = {
        'pick_top1_rate': diag_acc['top1_hits'] / steps_safe,
        'pick_top5_rate': diag_acc['top5_hits'] / steps_safe,
        'avg_er_rank': diag_acc['rank_sum'] / steps_safe,
        'mean_entropy': diag_acc['entropy_sum'] / steps_safe,
        'logit_er_corr': (diag_acc['corr_sum'] / max(1, diag_acc['corr_count'])) if diag_acc['corr_count'] > 0 else float('nan'),
    }
    return lam2_final, float(reward_log), total_steps, base_lam2, diag
  
# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    # Modes
    p.add_argument('--train', action='store_true')
    p.add_argument('--inference_only', action='store_true')

    # Graph/task
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--m', type=int, default=100)
    p.add_argument('--multi_task', action='store_true')
    p.add_argument('--n_min', type=int, default=16)
    p.add_argument('--n_max', type=int, default=32)
    p.add_argument('--dens_min', type=float, default=0.03)
    p.add_argument('--dens_max', type=float, default=1.0)
    # Multi-task bucketization and sampler controls
    p.add_argument('--n_buckets', type=int, default=4, help='Number of n-buckets for hardness-aware sampling')
    p.add_argument('--dens_buckets', type=int, default=6, help='Number of density buckets in [dens_min, dens_max]')
    p.add_argument('--sampler_alpha', type=float, default=0.5, help='Mixture weight of hardness-aware distribution vs uniform (0..1)')
    p.add_argument('--sampler_tau', type=float, default=0.5, help='Temperature for hardness weighting exp(-ma/tau)')
    p.add_argument('--sampler_warmup', type=int, default=100, help='Episodes of pure uniform sampling before hardness kicks in')
    p.add_argument('--sampler_smooth', type=float, default=0.1, help='EMA smoothing factor for per-bucket reward moving averages')
    p.add_argument('--init', type=str, default='path')

    # Heuristic pruning
    p.add_argument('--heuristic', type=str, default='er')
    p.add_argument('--topk', type=int, default=32)
    p.add_argument('--topk_frac', type=float, default=0.2)
    p.add_argument('--rand_cand_frac', type=float, default=0.0, help='Append this fraction of random non-ER candidate edges (0 to disable).')
    p.add_argument('--rand_cand_max', type=int, default=16, help='Max number of random candidates to append per step.')

    # Spectral
    p.add_argument('--spectral_backend', type=str, default='scipy', choices=['scipy','torch'])
    p.add_argument('--fast_spectral', action='store_true')
    p.add_argument('--spectral_refresh_k', type=int, default=1)

    # Model arch
    p.add_argument('--gat_hidden', type=int, default=64)
    p.add_argument('--gat_heads', type=int, default=6)
    p.add_argument('--gat_layers', type=int, default=3)
    p.add_argument('--edge_mlp_hidden', type=int, default=256)
    p.add_argument('--value_mlp_hidden', type=int, default=128)

    # PPO
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent_coef', type=float, default=0.02)
    p.add_argument('--vf_coef', type=float, default=0.6)
    p.add_argument('--ppo_epochs', type=int, default=8)
    p.add_argument('--clip_vf', type=float, default=0.2, help='Value function clipping range (0 to disable)')
    p.add_argument('--mb_size', type=int, default=4096)
    p.add_argument('--batch_episodes', type=int, default=8)
    p.add_argument('--train_temperature', type=float, default=1.0)
    # Endgame annealing to reduce oscillations
    p.add_argument('--anneal_endgame', action='store_true', help='Enable endgame annealing of lr/entropy/clip/temperature')
    p.add_argument('--anneal_start_frac', type=float, default=0.7, help='Fraction of training after which to start annealing')
    p.add_argument('--lr_end', type=float, default=5e-5, help='Final LR at the end of annealing window')
    p.add_argument('--clip_end', type=float, default=0.10, help='Final PPO clip at the end of annealing window')
    p.add_argument('--ent_end', type=float, default=0.005, help='Final entropy coef at the end of annealing window')
    p.add_argument('--temp_end', type=float, default=0.3, help='Final sampling temperature at the end of annealing window')
    # EMA/consistency stabilizers
    p.add_argument('--ema_policy', action='store_true', help='Use EMA teacher policy for data collection (behavior) after warmup')
    p.add_argument('--ema_start', type=int, default=0, help='Episode to start acting with EMA')
    p.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher network')
    p.add_argument('--consistency_coef', type=float, default=0.0, help='KL(student||EMA) regularizer weight (0 to disable)')
    p.add_argument('--consistency_decay', type=float, default=1.0, help='Multiplicative decay of consistency_coef after each PPO update')

    # Terminal reward vs ER baseline (logging & optional final-return assignment)
    p.add_argument('--reward_mode', type=str, default='margin', choices=['margin','ratio','logratio'],
                   help="How to compute terminal reward vs ER baseline: 'margin'=(model-ER), 'ratio'=(model-ER)/ER, 'logratio'=log(model)-log(ER)")
    p.add_argument('--reward_eps', type=float, default=1e-6, help='Small epsilon for ratio/logratio stability.')
    p.add_argument('--reward_clip', type=float, default=0.0, help='Absolute clip on reward; 0=off.')

    # Rollout shaping (per-step) — ratio only
    p.add_argument('--rollout_reward', action='store_true',
               help='Enable per-step shaped reward vs ER-top1 using ratio: (λA-λB)/max(eps,|λB|). If set, takes precedence over --final_return_reward.')
    p.add_argument('--rollout_reward_eps', type=float, default=1e-6,
               help='Stability epsilon for per-step ratio shaping.')
    p.add_argument('--rollout_reward_scale', type=float, default=1.0,
               help='Multiply each per-step shaped reward by this factor.')
    p.add_argument('--rollout_reward_clip', type=float, default=0.0,
               help='If >0, clip each per-step shaped reward to [-clip, +clip].')
    p.add_argument('--reg_shaping_coef', type=float, default=1,
                   help='Scale of per-step variance-reduction shaping when regularity is possible (set 0 to disable).')

    p.add_argument('--final_return_reward', action='store_true',
               help='Pure terminal reward: final λ2 minus ER-greedy baseline from the initial graph.')

    # PPO stabilizers
    p.add_argument('--target_kl', type=float, default=0.04, help='Early-stop PPO minibatches when approx KL exceeds this (x1.5) to prevent collapse.')
    p.add_argument('--anti_collapse', action='store_true', default=True, help='Enable LR/entropy scheduling and rollback-on-collapse guard')
    p.add_argument('--guard_window', type=int, default=64, help='Episodes window for moving-average collapse guard')
    p.add_argument('--guard_drop', type=float, default=0.05, help='Allowed drop in moving-average reward before triggering rollback')
    p.add_argument('--guard_patience', type=int, default=2, help='How many consecutive window drops before rollback')
    p.add_argument('--lr_min', type=float, default=5e-5, help='Lower bound for adaptive LR')
    p.add_argument('--aux_q_coef', type=float, default=0.5, help='Weight for auxiliary Q-head regression to terminal λ2 targets.')
    p.add_argument('--adv_clip', type=float, default=0.0, help='Clip normalized advantages to [-adv_clip,+adv_clip] (0=off)')

    # Runtime
    p.add_argument('--episodes', type=int, default=2000)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')

    # IO
    p.add_argument('--save_model', type=str, default='v11.pt')
    p.add_argument('--load_model', type=str, default='')
    p.add_argument('--save_every', type=int, default=200)
    p.add_argument('--save_ema_teacher', action='store_true', help='Save EMA teacher weights instead of student as primary state_dict')
    p.add_argument('--log_csv', type=str, default='', help='Optional CSV file to append per-episode metrics')
    p.add_argument('--train_log', type=str, default='train.log', help='Path to JSONL file for detailed per-step logs (empty to disable).')
    p.add_argument('--log_detail_every', type=int, default=1, help='Log detailed steps every N episodes.')
    p.add_argument('--log_max_steps', type=int, default=12, help='Max number of steps per episode to log in detail to limit file size.')

    args = p.parse_args()

    set_num_threads(args.threads)
    set_seed(args.seed)

    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')

    # Build net/agent
    arch = {
        'node_in': 5,
        'hid': args.gat_hidden,
        'heads': args.gat_heads,
        'layers': args.gat_layers,
        'edge_mlp_hidden': args.edge_mlp_hidden,
        'value_mlp_hidden': args.value_mlp_hidden,
        'buckets': args.dens_buckets
    }

    net = PolicyValueNet(arch['node_in'], arch['hid'], arch['heads'], arch['layers'], arch['edge_mlp_hidden'], arch['value_mlp_hidden'],device=device.type).to(device) 
    ppo_cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        ppo_epochs=args.ppo_epochs, mb_size=args.mb_size, train_temperature=args.train_temperature,
        target_kl=args.target_kl, clip_vf=args.clip_vf, lr_min=args.lr_min, aux_q_coef=args.aux_q_coef,
        ema_decay=args.ema_decay, consistency_coef=args.consistency_coef, consistency_decay=args.consistency_decay,
        adv_clip=args.adv_clip
    )
    agent = PPOAgent(net, ppo_cfg, device=device)

    if args.load_model:
        obj = agent.load(args.load_model, device=device)
        chk_arch = obj.get('arch', arch)

        # Rebuild net/agent if arch differs
        if json.dumps(chk_arch, sort_keys=True) != json.dumps(arch, sort_keys=True):
            arch = chk_arch
            net = PolicyValueNet(
                arch['node_in'], arch['hid'], arch['heads'], arch['layers'],
                arch['edge_mlp_hidden'], arch['value_mlp_hidden'],
                device=device.type # CHANGED
            ).to(device)
            agent = PPOAgent(net, ppo_cfg, device=device)

        # Load weights/opt
        net.load_state_dict(obj['state_dict'], strict=True)
        # Load EMA teacher if present; default to same weights if absent
        try:
            agent.ema_net.load_state_dict(obj.get('ema_state_dict', obj['state_dict']))
        except Exception:
            pass
        try:
            agent.opt.load_state_dict(obj['opt'])
        except Exception:
            pass
        print(f"[load] Loaded model from: {args.load_model}")

        # --- Sync CLI args to checkpoint to avoid bucket mismatches later ---
        try:
            args.dens_buckets = int(arch.get('buckets', args.dens_buckets))
        except Exception:
            pass

    # --- Hardness-aware task sampler state ---
    Bn = max(1, int(args.n_buckets)); Bd = max(1, int(args.dens_buckets)); B = Bn * Bd
    bucket_ma = np.zeros(B, dtype=np.float64)
    bucket_cnt = np.zeros(B, dtype=np.int64)

    def _bucket_bounds():
        # integer size buckets
        span = max(1, args.n_max - args.n_min + 1)
        width = int(math.ceil(span / Bn))
        n_bounds = []
        for bi in range(Bn):
            lo = args.n_min + bi * width
            hi = min(args.n_max, lo + width - 1)
            n_bounds.append((lo, hi))
        # density buckets in [dens_min, dens_max]
        d_bounds = []
        for bj in range(Bd):
            lo = args.dens_min + (args.dens_max - args.dens_min) * (bj / Bd)
            hi = args.dens_min + (args.dens_max - args.dens_min) * ((bj+1) / Bd)
            lo = max(args.dens_min, min(args.dens_max, lo))
            hi = max(args.dens_min, min(args.dens_max, hi))
            d_bounds.append((lo, hi))
        return n_bounds, d_bounds

    n_bounds, d_bounds = _bucket_bounds()

    def _bucket_prob(ep_idx:int):
        # mixture between uniform and hardness-aware softmax over -ma
        alpha = float(args.sampler_alpha) if ep_idx >= int(args.sampler_warmup) else 0.0
        if alpha <= 0.0:
            return np.ones(B, dtype=np.float64) / B
        x = -bucket_ma / max(1e-6, float(args.sampler_tau))
        x = x - x.max()  # stable
        w = np.exp(x)
        w = w / max(1e-12, w.sum())
        u = np.ones(B, dtype=np.float64) / B
        p = (1.0 - alpha) * u + alpha * w
        # renorm to be safe
        p = p / max(1e-12, p.sum())
        return p

    def _sample_from_bucket(bidx:int) -> Tuple[int,int]:
        bi = bidx // Bd; bj = bidx % Bd
        n_lo, n_hi = n_bounds[bi]
        # sample n uniformly in bucket bounds
        n = random.randint(n_lo, n_hi)
        # sample density in bucket bounds
        d_lo, d_hi = d_bounds[bj]
        t = random.uniform(d_lo, d_hi)
        # map to m
        Mmax = n * (n - 1) // 2
        base = n - 1
        m = int(round(base + t * (Mmax - base)))
        m = max(base, min(Mmax, m))
        return n, m

    def sample_task(ep_idx:int):
        if not args.multi_task:
            return args.n, args.m, 0
        p = _bucket_prob(ep_idx)
        b = int(np.random.choice(np.arange(B), p=p))
        n, m = _sample_from_bucket(b)
        return n, m, b

    if args.inference_only:
        n, m = args.n, args.m
        cfg = EnvConfig(n=n, m=m, init=args.init, spectral_backend=args.spectral_backend, device=device.type, fast_spectral=False, spectral_refresh_k=args.spectral_refresh_k)
        env = GraphBuildEnv(cfg)
        # Allow using EMA teacher for inference via flag
        if getattr(args, 'ema_policy', False):
            setattr(args, 'ema_infer', True)
        lam2, reward, steps, base_lam2, diag = run_episode(env, net, agent, args, train=False, ep=0)
        # Determine if the final graph is regular (all degrees equal)
        deg = np.rint(env.adj.sum(axis=1)).astype(int)
        is_reg = (deg.min() == deg.max())
        reg_k = int(deg[0]) if (is_reg and deg.size > 0) else None
        print(f"=== Inference-only ===")
        print(f"Heuristic=ER TopK={args.topk}  greedy=True  n={args.n} m={args.m}")
        print(f"λ2 ≈ {lam2:.6f}  base≈{base_lam2:.6f}  reward={reward:+.6f} ({args.reward_mode})  steps={steps}")
        print(f"pick@1={diag['pick_top1_rate']:.2f}  pick@5={diag['pick_top5_rate']:.2f}  avgERrank={diag['avg_er_rank']:.1f}  corr(logit,ER)={diag['logit_er_corr']:.2f}  H={diag['mean_entropy']:.2f}")
        print(f"regular={is_reg}" + (f" k={reg_k}" if reg_k is not None else ""))
        sys.exit(0)

    # Training loop
    # Report intended spectral settings for training (per-episode env may override via rollout shaping)
    print(f"[spectral] backend={args.spectral_backend} fast_suggested={args.fast_spectral} refresh_k_suggested={args.spectral_refresh_k} rollout_reward={args.rollout_reward} rollout_mode=ratio")
    # --- Anti-collapse guard state ---
    recent_rewards = deque(maxlen=args.guard_window)
    best_ma = -1e9
    best_state = copy.deepcopy(agent.net.state_dict())
    best_opt   = copy.deepcopy(agent.opt.state_dict())
    guard_strikes = 0
    # --- Endgame anneal baselines ---
    anneal = {
        'enabled': bool(getattr(args, 'anneal_endgame', False)),
        'start_ep': int(math.floor(args.episodes * max(0.0, min(0.99, float(getattr(args, 'anneal_start_frac', 0.7)))))),
        'lr0': float(agent.opt.param_groups[0]['lr']),
        'clip0': float(agent.cfg.clip),
        'ent0': float(agent.cfg.ent_coef),
        'temp0': float(args.train_temperature),
        'lr1': float(getattr(args, 'lr_end', args.lr_min)),
        'clip1': float(getattr(args, 'clip_end', 0.10)),
        'ent1': float(getattr(args, 'ent_end', 0.005)),
        'temp1': float(getattr(args, 'temp_end', 0.3)),
    }
    t0=time.time()
    for ep in range(1, args.episodes+1):
        n, m, bidx = sample_task(ep)
        # Use fast spectral only when NOT using rollout-shaped rewards.
        # With rollout shaping we need exact, fresh spectral features each step for low-variance credit.
        use_fast = (args.fast_spectral and (not args.rollout_reward))
        refresh_k = 1 if args.rollout_reward else args.spectral_refresh_k
        cfg = EnvConfig(
            n=n, m=m, init=args.init,
            spectral_backend=args.spectral_backend,
            device=device.type,
            fast_spectral=use_fast,
            spectral_refresh_k=refresh_k
        )
        env = GraphBuildEnv(cfg)
        lam2, reward, steps, base_lam2, diag = run_episode(env, net, agent, args, train=True, ep=ep)
        dens = normalized_density(n, m)

        # Update per-bucket moving average
        if 0 <= bidx < B:
            beta = float(args.sampler_smooth)
            bucket_ma[bidx] = (1.0 - beta) * bucket_ma[bidx] + beta * float(reward)
            bucket_cnt[bidx] += 1

        # buffer flush policy: update every batch_episodes
        if (ep % args.batch_episodes)==0:
            stats = agent.ppo_update()
            if args.anti_collapse:
                cur_lr = agent.opt.param_groups[0]['lr']
                if stats['kl'] > args.target_kl * 1.5:
                    new_lr = max(agent.cfg.lr_min, cur_lr * 0.7)
                    agent.opt.param_groups[0]['lr'] = new_lr
                    agent.cfg.ent_coef = agent.cfg.ent_coef * 1.05
                elif stats['kl'] < args.target_kl * 0.5:
                    new_lr = min(args.lr, cur_lr * 1.05)
                    agent.opt.param_groups[0]['lr'] = new_lr
            # Endgame annealing to damp oscillations
            if anneal['enabled'] and ep >= anneal['start_ep']:
                t = (ep - anneal['start_ep']) / max(1, args.episodes - anneal['start_ep'])
                t = min(1.0, max(0.0, t))
                # Only decrease LR from current value towards lr1 (monotone non-increasing)
                lr_target = anneal['lr0'] + (anneal['lr1'] - anneal['lr0']) * t
                cur_lr = agent.opt.param_groups[0]['lr']
                agent.opt.param_groups[0]['lr'] = min(cur_lr, lr_target)
                # Linearly anneal clip, entropy, and temperature
                agent.cfg.clip = anneal['clip0'] + (anneal['clip1'] - anneal['clip0']) * t
                agent.cfg.ent_coef = anneal['ent0'] + (anneal['ent1'] - anneal['ent0']) * t
                args.train_temperature = anneal['temp0'] + (anneal['temp1'] - anneal['temp0']) * t
            print(
                f"[ep {ep:5d}] n={n:2d} m={m:4d} dens={dens:.2f} λ2={lam2:.3f} base={base_lam2:.3f} reward={reward:+.3f} steps={steps:3d} | upd loss={stats['loss']:.4f} pol={stats['pol']:.4f} val={stats['val']:.4f} q={stats['q']:.4f} ent={stats['ent']:.4f} kl={stats['kl']:.4f} clipfrac={stats['clipfrac']:.2f} lr={agent.opt.param_groups[0]['lr']:.2e} ent_coef={agent.cfg.ent_coef:.3f}"
                + f" pick@1={diag['pick_top1_rate']:.2f} pick@5={diag['pick_top5_rate']:.2f} avgERrank={diag['avg_er_rank']:.1f} corr={diag['logit_er_corr']:.2f} H={diag['mean_entropy']:.2f}"
                + f" bucket={bidx}"
            )
        else:
            print(
                f"[ep {ep:5d}] n={n:2d} m={m:4d} dens={dens:.2f} λ2={lam2:.3f} base={base_lam2:.3f} reward={reward:+.3f} steps={steps:3d} | no-update lr={agent.opt.param_groups[0]['lr']:.2e} ent_coef={agent.cfg.ent_coef:.3f}"
                + f" pick@1={diag['pick_top1_rate']:.2f} pick@5={diag['pick_top5_rate']:.2f} avgERrank={diag['avg_er_rank']:.1f} corr={diag['logit_er_corr']:.2f} H={diag['mean_entropy']:.2f}"
                + f" bucket={bidx}"
            )

        # Collapse guard logic
        if args.anti_collapse:
            recent_rewards.append(reward)
            if len(recent_rewards) == recent_rewards.maxlen:
                ma = float(np.mean(recent_rewards))
                if ma > best_ma + 1e-12:
                    best_ma = ma
                    best_state = copy.deepcopy(agent.net.state_dict())
                    best_opt   = copy.deepcopy(agent.opt.state_dict())
                    guard_strikes = 0
                elif ma < (best_ma - args.guard_drop):
                    guard_strikes += 1
                    print(f"[guard] moving-avg reward dropped to {ma:+.4f} (< best {best_ma:+.4f} - {args.guard_drop}); strike {guard_strikes}/{args.guard_patience}")
                    if guard_strikes >= args.guard_patience:
                        print("[guard] rollback to best checkpoint and dampen LR; boosting entropy a bit")
                        agent.net.load_state_dict(best_state)
                        try:
                            agent.opt.load_state_dict(best_opt)
                        except Exception:
                            pass
                        cur_lr = agent.opt.param_groups[0]['lr']
                        agent.opt.param_groups[0]['lr'] = max(agent.cfg.lr_min, cur_lr * 0.7)
                        agent.cfg.ent_coef = agent.cfg.ent_coef * 1.10
                        guard_strikes = 0

        # Append CSV log if requested
        if args.log_csv:
            header = ["ep","n","m","dens","lam2","base","reward","steps","updated",
                      "loss","pol","val","q","ent","kl","clipfrac",
                      "pick_top1","pick_top5","avg_er_rank","corr_logit_er","mean_entropy","bucket"]
            if (ep % args.batch_episodes)==0:
                # we just computed stats
                log_row(args.log_csv, header, [
                    ep, n, m, f"{dens:.6f}", f"{lam2:.6f}", f"{base_lam2:.6f}", f"{reward:.6f}", steps,
                    1, f"{stats['loss']:.6f}", f"{stats['pol']:.6f}", f"{stats['val']:.6f}", f"{stats['q']:.6f}",
                    f"{stats['ent']:.6f}", f"{stats['kl']:.6f}", f"{stats['clipfrac']:.6f}",
                    f"{diag['pick_top1_rate']:.6f}", f"{diag['pick_top5_rate']:.6f}",
                    f"{diag['avg_er_rank']:.6f}", f"{diag['logit_er_corr']:.6f}", f"{diag['mean_entropy']:.6f}",
                    int(bidx)
                ])
            else:
                log_row(args.log_csv, header, [
                    ep, n, m, f"{dens:.6f}", f"{lam2:.6f}", f"{base_lam2:.6f}", f"{reward:.6f}", steps,
                    0, "", "", "", "", "", "",
                    f"{diag['pick_top1_rate']:.6f}", f"{diag['pick_top5_rate']:.6f}",
                    f"{diag['avg_er_rank']:.6f}", f"{diag['logit_er_corr']:.6f}", f"{diag['mean_entropy']:.6f}",
                    int(bidx)
                ])

        if args.save_every>0 and (ep % args.save_every)==0:
            path_ep = episodic_path(args.save_model, ep)
            agent.save(path_ep, arch, prefer_ema=bool(getattr(args, 'save_ema_teacher', False)))
            print(f"[save] -> {path_ep}")

    agent.save(args.save_model, arch, prefer_ema=bool(getattr(args, 'save_ema_teacher', False)))
    print(f"[done] saved -> {args.save_model} | elapsed={time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()




