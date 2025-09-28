"""
python3 src/train_phase1.py --infer --load_model models_phase1/phase1_max32_rl_ep01100.pt --n 48 --m 264 --topk 16 --device cpu
"""
import os
import re
import math
import argparse
import random
from typing import List, Tuple, Dict

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


def er_topk_from_G(adj: np.ndarray, G: np.ndarray, diagG: np.ndarray, k: int, frac_cap: float = 0.0):
    n = adj.shape[0]
    iu, iv = np.triu_indices(n, k=1)
    mask_non = (adj[iu, iv] == 0)
    iu = iu[mask_non]; iv = iv[mask_non]
    if iu.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    er = diagG[iu] + diagG[iv] - 2.0 * G[iu, iv]
    total = len(er)
    k_eff = k
    if frac_cap > 0:
        cap = int(math.ceil(frac_cap * total))
        k_eff = min(k, cap)
    kk = min(k_eff, total)
    if kk <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    idx = np.argpartition(-er, kk - 1)[:kk]
    order = np.argsort(-er[idx])
    idx = idx[order]
    return iu[idx], iv[idx], er[idx]


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
        self.G = None
        self.diagG = None
        self.use_green = False
        self.t = 0
        return self.state()

    def state(self):
        n, k = self.n, self.k
        e_now = int(self.adj.sum() // 2)
        progress = float(e_now / max(1, self.m))
        # minimal rotation-invariant node features
        deg = self.adj.sum(axis=1).astype(np.float32)
        degn = deg / max(1.0, float(deg.max()))
        diagG = self.diagG if (self.diagG is not None) else np.zeros(n, dtype=np.float64)
        node_feats = np.stack([
            degn.astype(np.float32),
            diagG.astype(np.float32),  # Green diag (resistance centrality)
        ], axis=1)
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

    def candidates(self, topk: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            uu = uu[mask_cross]; vv = vv[mask_cross]
            if uu.size == 0:
                return uu, vv, np.zeros(0, dtype=np.float64)
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

        # score and select topk
        if self.use_green and self.G is not None and self.diagG is not None:
            ers = (self.diagG[uu] + self.diagG[vv] - 2.0 * self.G[uu, vv])
            score = ers
        else:
            defc = (self.k - deg)
            score = defc[uu] + defc[vv]
        if uu.size > topk:
            idx = np.argpartition(-score, topk - 1)[:topk]
            order = np.argsort(-score[idx])
            idx = idx[order]
            uu, vv = uu[idx], vv[idx]
            if 'ers' in locals():
                ers = ers[idx]
            else:
                ers = np.zeros_like(uu, dtype=np.float64)
        else:
            if 'ers' not in locals():
                ers = np.zeros_like(uu, dtype=np.float64)
        return uu, vv, ers

    def step(self, u: int, v: int):
        # place edge
        if u != v and self.adj[u, v] == 0.0:
            self.adj[u, v] = 1.0; self.adj[v, u] = 1.0
            # update Green if enabled
            if self.use_green and (self.G is not None):
                self.G, self.diagG = G_rank1_update_edge(self.G, self.diagG, u, v, 1.0)
            else:
                # switch on Green when connected
                if is_connected_adj(self.adj):
                    try:
                        self.G, self.diagG = G_from_adj(self.adj)
                        self.use_green = True
                    except Exception:
                        self.use_green = False
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
    # ER if available
    if env.use_green and env.G is not None and env.diagG is not None:
        ers = (env.diagG[uu] + env.diagG[vv] - 2.0 * env.G[uu, vv]).astype(np.float32)
    else:
        ers = np.zeros_like(du, dtype=np.float32)
    # cross-component indicator
    comp = env._components()
    cross = (comp[uu] != comp[vv]).astype(np.float32)

    edge_feats = np.stack([
        du, dv,
        np.full_like(du, progress),
        np.full_like(du, dens_target),
        cross,
        ers,
        def_u, def_v,
        dVar, overlap,
        np.full_like(du, float(np.mean(np.maximum(0.0, k - deg)))),
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
    def __init__(self, node_in=2, edge_in=11, global_in=5, hid=32, heads=4, layers=2, edge_hidden=256):
        super().__init__()
        self.enc = GATEncoder(node_in, hid, heads, layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.enc.out_dim * 2 + edge_in + global_in, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

    def forward(self, node_feats: torch.Tensor, adj_dense: torch.Tensor, cand_edges: torch.Tensor,
                edge_feats: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        x = self.enc(node_feats, adj_dense)
        if cand_edges.numel() == 0:
            return torch.empty(0, device=x.device)
        u = cand_edges[:, 0]; v = cand_edges[:, 1]
        xu = x[u]; xv = x[v]
        uv = torch.cat([xu, xv], dim=1)
        gf = global_feats.unsqueeze(0).expand(uv.size(0), -1)
        full = torch.cat([uv, edge_feats, gf], dim=1)
        logits = self.edge_mlp(full).squeeze(-1)
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

def sample_task(n_min: int, n_max: int) -> Tuple[int, int]:
    n = random.randint(n_min, n_max)
    # choose k in [2, n-2] such that n*k is even
    ks = [k for k in range(2, max(2, n - 1)) if ((n * k) % 2 == 0)]
    if not ks:
        return n, 2
    k = random.choice(ks)
    return n, k


def final_reward(env: Phase1Env) -> float:
    n, k = env.n, env.k
    A = env.adj
    deg = A.sum(axis=1)
    reg = float(np.allclose(deg, k, atol=1e-6))
    conn = float(is_connected_adj(A))
    # encourage high λ2 among regular graphs
    L = laplacian_from_adj(A)
    w, _ = np.linalg.eigh(L)
    w = np.sort(w)
    lam2 = float(w[1]) if len(w) > 1 else 0.0
    # composite: large bonus for exact regular+connected, stronger λ2 weight
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
    return -0.02 * dF + bonus_conn

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
    topk: int = 8,
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
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pol = EdgePolicyNet(node_in=2, edge_in=11, global_in=5, hid=64, heads=6, layers=3, edge_hidden=384).to(device)
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

    for ep in range(1, episodes + 1):
        n, k = sample_task(n_min, n_max)
        # avoid trivial extremes
        if k <= 1 or k >= n - 1:
            continue
        env = Phase1Env(n, k)
        traj = []  # list of (node, adj, cand, edge, glob, glob_next, act, logp_old, v_old, r, done)
        done = False
        while not done:
            st = env.state()
            glob = to_t(st['global'])
            iu, iv, _ = env.candidates(topk=topk)
            if iu.size == 0:
                break
            e_feats_np = edge_features(env, iu, iv)
            e_feats = to_t(e_feats_np)
            # tensors for GNN
            node_feats = to_t(env.state()['node_feats'])
            adj_dense = to_t(env.adj)
            cand = torch.as_tensor(np.stack([iu, iv], axis=1), dtype=torch.long, device=device)
            logits = pol(node_feats, adj_dense, cand, e_feats, glob) / temp
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp_old = dist.log_prob(a).detach()
            v_old = val(glob).detach()
            a_i = int(a.detach().cpu().item())
            u = int(iu[a_i]); v = int(iv[a_i])
            # stepwise shaping: encourage ER closeness to top1 when connected, weighted by deficit progression
            ers_vec = e_feats_np[:, 5]
            mean_def = float(np.mean(np.maximum(0.0, env.k - env.adj.sum(axis=1))))
            w_spec = max(0.0, min(1.0, 1.0 - (mean_def / max(1.0, env.k))))
            if ers_vec.size > 0:
                top1 = float(ers_vec.max())
                er_rel = float(ers_vec[a_i] / max(1e-8, top1))
            else:
                er_rel = 0.0
            r = step_reward(env, u, v) + (0.2 * w_spec) * er_rel
            st2, done = env.step(u, v)
            glob_next = to_t(st2['global'])
            traj.append((
                node_feats.detach(), adj_dense.detach(), cand.detach(), e_feats.detach(),
                glob.detach(), glob_next.detach(), a.detach(), logp_old, v_old, r, done
            ))

        # add terminal bonus to last step reward
        if traj:
            Gt = final_reward(env)
            last = list(traj[-1])
            last[-2] = last[-2]  # v_old unchanged
            last[-3] = last[-3]  # logp_old unchanged
            last[-1] = last[-1] + Gt  # reward += terminal bonus
            last[-0] = last[-0]  # done flag
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
                    for i in mb:
                        node = node_list[i].to(device)
                        adj = adj_list[i].to(device)
                        cand = cand_list[i].to(device)
                        edgef = edge_list[i].to(device)
                        glob = glob_list[i].to(device)
                        act = act_list[i].to(device)
                        logits = pol(node, adj, cand, edgef, glob)
                        dist = Categorical(logits=logits)
                        logp_news.append(dist.log_prob(act))
                        ents.append(dist.entropy().mean())
                        v_news.append(val(glob))
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
                    v_loss = vf_coef * F.mse_loss(v_news, ret_mb)
                    loss = pol_loss + v_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            buffer.clear(); ep_in_batch = 0

        ema_return = 0.95 * ema_return + 0.05 * float(Gt)
        if ep % 10 == 0:
            print(f"[ep {ep}] n={n} k={k} steps={len(traj)} Gt={Gt:.3f} emaR={ema_return:.3f}")

        if eval_every > 0 and (ep % eval_every) == 0:
            eval_on_enum(pol, n_min, n_max, data_dir, eval_samples, topk, device)

        # periodic checkpointing
        if save_every > 0 and (ep % save_every) == 0:
            base = os.path.splitext(save_path)[0]
            ckpt_path = f"{base}_ep{ep:05d}.pt"
            try:
                torch.save({'policy': pol.state_dict(), 'value': val.state_dict(), 'opt': opt.state_dict()}, ckpt_path)
                print(f"[save] -> {ckpt_path}")
            except Exception as e:
                print(f"[save] Warning: failed to save checkpoint {ckpt_path}: {e}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({'policy': pol.state_dict(), 'value': val.state_dict(), 'opt': opt.state_dict()}, save_path)
    print(f"[phase1-rl] saved -> {save_path}")


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


@torch.no_grad()
def eval_on_enum(pol: EdgePolicyNet, n_min: int, n_max: int, data_dir: str, samples: int, topk: int, device: str):
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
            iu, iv, _ = env.candidates(topk=topk)
            if iu.size == 0:
                break
            edges_left = env.m - int(env.adj.sum() // 2)
            if edges_left <= n:
                # exact lookahead near the end
                best_idx = 0; best_lam2 = -1e9
                for ci in range(len(iu)):
                    u = int(iu[ci]); v = int(iv[ci])
                    lam2_val = _lam2_after_add_and_regular_fill(env, u, v)
                    if lam2_val > best_lam2:
                        best_lam2 = lam2_val; best_idx = ci
                env.step(int(iu[best_idx]), int(iv[best_idx]))
            else:
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
            A_imp, lam2_imp = _two_switch_improve_lambda2(env.adj.copy(), k, max_iters=50, tries_per_iter=50)
            lam2_val = lam2_imp
        else:
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
def infer_regular(load_path: str, n: int, m: int, topk: int = 8, device: str = 'cpu'):
    if (2 * m) % n != 0:
        print(f"[infer] (n={n}, m={m}) not regularizable: 2m % n != 0")
        return
    k = (2 * m) // n
    pol = EdgePolicyNet(node_in=2, edge_in=11, global_in=5, hid=64, heads=6, layers=3, edge_hidden=384).to(device)
    obj = torch.load(load_path, map_location=device)
    pol.load_state_dict(obj['policy'] if 'policy' in obj else obj)
    pol.eval()
    env = Phase1Env(n, k)
    steps = 0
    while not env.done() and steps < env.m + n:
        st = env.state()
        glob = torch.as_tensor(st['global'], dtype=torch.float32, device=device)
        iu, iv, _ = env.candidates(topk=topk)
        if iu.size == 0:
            break
        # Lookahead near the end: pick candidate maximizing final λ2 after greedy regular completion
        edges_left = env.m - int(env.adj.sum() // 2)
        if edges_left <= n:
            best_idx = 0; best_lam2 = -1e9
            for ci in range(len(iu)):
                u = int(iu[ci]); v = int(iv[ci])
                lam2_val = _lam2_after_add_and_regular_fill(env, u, v)
                if lam2_val > best_lam2:
                    best_lam2 = lam2_val; best_idx = ci
            env.step(int(iu[best_idx]), int(iv[best_idx]))
        else:
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
    print(f"[infer] n={n} m={m} k={k} regular={is_reg} lam2={lam2:.6f}")
    for i in range(n):
        nbrs = [str(j) for j in range(n) if env.adj[i, j] > 0.5]
        print(f"{i}: "+', '.join(nbrs))


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


def main():
    ap = argparse.ArgumentParser(description='Phase‑1 RL for regular graph construction (independent)')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--episodes', type=int, default=5000)
    ap.add_argument('--n_min', type=int, default=8)
    ap.add_argument('--n_max', type=int, default=32)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--data_dir', type=str, default='data/data_ENUM')
    ap.add_argument('--eval_samples', type=int, default=20)
    ap.add_argument('--save_model', type=str, default='models_phase1/p1.pt')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0, help='Set OMP/MKL/OPENBLAS threads and torch.set_num_threads')
    ap.add_argument('--ppo_mbsize', type=int, default=64)
    ap.add_argument('--refine_iters', type=int, default=100)
    ap.add_argument('--refine_tries', type=int, default=100)
    ap.add_argument('--save_every', type=int, default=100, help='Save checkpoint every N episodes (0 to disable)')
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
            topk=args.topk,
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
        )
    elif args.infer:
        if not args.load_model or args.n <= 0 or args.m <= 0:
            raise SystemExit('--infer requires --load_model, --n, --m')
        infer_regular(args.load_model, args.n, args.m, topk=args.topk, device=args.device)
    else:
        print('Specify --train or --infer')


if __name__ == '__main__':
    main()

