#!/usr/bin/env python3
# v9.py — Hard-bucketed (density) ER-pruned PPO with fast spectral caching
# Minimal, fast, single-file training/inference.
# - Per-bucket heads (edge/value) with shared GAT encoder
# - ER Top-K pruning
# - Final-return reward (sparse) with ER baseline margin
# - Fast spectral: skip per-step λ2; refresh every K steps; exact at episode end
# - Small, dependency-light (numpy, scipy, torch)

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

# JSONL logger for detailed per-step diagnostics
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

# Algebraic connectivity (λ2) + (optionally) Fiedler/next vectors via SciPy or torch

def lam2_with_vectors(L: np.ndarray, backend: str = "scipy", device: str = "cpu"):
    """
    Return (λ2, φ2, φ3) of the Laplacian L.
    SciPy path:
      * For small graphs (n ≤ 256), use dense eigh (robust, fast enough).
      * Otherwise, use eigsh with which='SM' (smallest magnitude) to avoid
        shift-invert at sigma=0 on a singular Laplacian. Fall back to dense eigh
        if ARPACK fails or returns NaNs/Infs.
    Torch path:
      * Use full eigh (CPU/GPU) for small graphs.
    """
    n = L.shape[0]
    if backend == "scipy":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy not available; use --spectral_backend torch")
        try:
            if n <= 256:
                # Dense is very stable for small n and avoids shift-invert pitfalls.
                w, v = np.linalg.eigh(L)
            else:
                # Use 'SM' (smallest magnitude) to get [0, λ2, λ3, ...] without shift-invert.
                # Work with a CSR to help ARPACK and set a mild tolerance.
                w, v = eigsh(csr_matrix(L), k=min(3, n-1), which='SM', tol=1e-6, maxiter=max(1000, 10*n))
        except Exception:
            # Robust fallback: dense eigh
            w, v = np.linalg.eigh(L)

        # If anything went numerically wrong, fall back to dense.
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(v)):
            w, v = np.linalg.eigh(L)

        idx = np.argsort(w)
        w, v = w[idx], v[:, idx]
        lam2 = float(w[1]) if len(w) > 1 else 0.0
        phi2 = v[:, 1] if v.shape[1] > 1 else np.zeros(n)
        phi3 = v[:, 2] if v.shape[1] > 2 else np.zeros(n)
        return lam2, phi2, phi3
    else:
        # torch backend (CPU/GPU)
        tL = torch.as_tensor(L, dtype=torch.float64, device=device)
        w, v = torch.linalg.eigh(tL)
        w = w.cpu().numpy(); v = v.cpu().numpy()
        idx = np.argsort(w)
        w, v = w[idx], v[:, idx]
        lam2 = float(w[1]) if len(w) > 1 else 0.0
        phi2 = v[:, 1] if v.shape[1] > 1 else np.zeros(L.shape[0])
        phi3 = v[:, 2] if v.shape[1] > 2 else np.zeros(L.shape[0])
        return lam2, phi2, phi3

# Effective resistance top-K via dense eig (n<=64 practical). For speed, we
# compute full pseudoinverse once per step.

def er_topk(n:int, adj: np.ndarray, k: int, frac_cap: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    # Build Laplacian and pseudoinverse L^+
    L = laplacian_from_adj(adj)
    # Dense eig (tiny n) — robust and fast enough here
    w, V = np.linalg.eigh(L)
    # Build L^+ via eigen decomposition (exclude zero eigen)
    tol = 1e-12
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    Lplus = (V * invw) @ V.T  # V diag(invw) V^T

    # Candidate list: all non-edges u<v
    iu, iv = np.triu_indices(n, k=1)
    mask_non = (adj[iu, iv] == 0)
    iu, iv = iu[mask_non], iv[mask_non]
    # ER(u,v) = L+_uu + L+_vv - 2L+_uv
    er = Lplus[iu, iu] + Lplus[iv, iv] - 2.0 * Lplus[iu, iv]

    # Top-K (with fraction cap)
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


# -----------------------------
# Greedy ER baseline (λ2 after greedy ER completion)
# -----------------------------
def er_greedy_baseline_lambda2(n: int, adj: np.ndarray, m: int, backend: str = 'scipy', device: str = 'cpu') -> float:
    """
    Greedy ER completion baseline: starting from `adj`, repeatedly add the edge
    with the largest effective resistance (Top-1) until the graph has `m` edges.
    Finally compute and return λ2 of the completed graph.
    Notes:
      * We never compute λ2 during the rollout, only once at the end.
      * Each ER step needs a pseudoinverse (eigendecomp) – OK for n<=~64.
    """
    adj_b = adj.copy()
    base_edges = int(adj_b.sum() // 2)
    steps = max(0, m - base_edges)
    for _ in range(steps):
        iu, iv, er = er_topk(n, adj_b, k=1, frac_cap=1.0)
        if len(iu) == 0:
            break
        u = int(iu[0]); v = int(iv[0])
        if adj_b[u, v] == 0.0:
            adj_b[u, v] = 1.0
            adj_b[v, u] = 1.0
    Lb = laplacian_from_adj(adj_b)
    lam2_b, _, _ = lam2_with_vectors(Lb, backend=backend, device=device)
    return lam2_b

def lam2_after_add_and_fill(n: int, adj: np.ndarray, m: int, u: int, v: int,
                            backend: str = 'scipy', device: str = 'cpu') -> float:
    """
    Add (u,v) to a copy of adj, then greedily complete with ER Top-1 until m edges.
    Return final λ2. Used for rollout-shaped rewards.
    """
    adj2 = adj.copy()
    if u != v and adj2[u, v] == 0.0:
        adj2[u, v] = 1.0
        adj2[v, u] = 1.0
    return er_greedy_baseline_lambda2(n, adj2, m, backend=backend, device=device)

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
    spectral_refresh_k: int = 6

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
# Per-bucket isolated subnets (20 by default)
# Each bucket has its own encoder + heads: no cross-density influence.
# -----------------------------

class BucketNet(nn.Module):
    def __init__(self, node_in, hid, heads, layers, edge_hidden, value_hidden, bucket_emb_dim):
        super().__init__()
        # Private encoder for this bucket (cold wall preserved)
        self.encoder = GATEncoder(node_in, hid, heads, layers)
        self.node_emb_dim = self.encoder.out_dim

        # Global graph-context attention pooling (learnable query)
        self.gc_query = nn.Parameter(torch.randn(self.node_emb_dim) * 0.01)

        # Learnable bucket embedding (kept for parity with v8)
        self.bucket_emb = nn.Parameter(torch.randn(bucket_emb_dim) * 0.01)
        self.bucket_emb_dim = bucket_emb_dim

        # Edge scorer now sees: [H_u || H_v || gc || edge_feat_dim edge feats || 4 global feats || bucket emb]
        edge_in = self.node_emb_dim * 2 + self.node_emb_dim + 7 + 4 + bucket_emb_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        # Q-head predicts terminal λ2 after adding the candidate and greedy-completing (used as auxiliary loss)
        self.q_mlp = nn.Sequential(
            nn.Linear(edge_in, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        # Value head: mean(H) + [log1p(n), dens_target, progress, λ2] + bucket emb
        value_in = self.node_emb_dim + 4 + bucket_emb_dim
        self.value_mlp = nn.Sequential(
            nn.Linear(value_in, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, node_feats, adj_dense, cand_edges, edge_feats_7, global_feats_4):
        """
        node_feats:    (n, F)
        adj_dense:     (n, n)
        cand_edges:    (K, 2) long
        edge_feats_7:    (K, 7)
        global_feats_4:(4,)   -> [log1p(n), dens_target, progress, λ2]
        """
        x = self.encoder(node_feats, adj_dense)  # (n, D)

        # Value head uses pooled node embeddings + global features + bucket embedding
        pool = x.mean(dim=0)  # (D,)
        gfull = torch.cat([pool, global_feats_4, self.bucket_emb], dim=0)
        value = self.value_mlp(gfull).squeeze(-1)

        # If no candidates, return empty logits/q and the value
        if cand_edges.numel() == 0:
            return torch.empty(0, device=x.device), value, torch.empty(0, device=x.device)

        # -------- Global graph-context pooling (attention with learnable query) --------
        # scores_i = H_i · q, softmax over nodes, context = Σ_i α_i H_i
        scores = torch.matmul(x, self.gc_query)              # (n,)
        alpha = torch.softmax(scores, dim=0).unsqueeze(1)    # (n,1)
        gc = (alpha * x).sum(dim=0)                          # (D,)

        # Candidate-specific tensors
        u = cand_edges[:, 0]
        v = cand_edges[:, 1]
        xu = x[u]                               # (K, D)
        xv = x[v]                               # (K, D)
        uv = torch.cat([xu, xv], dim=1)         # (K, 2D)

        # Replicate global context and features per-candidate
        K = cand_edges.size(0)
        gc_rep = gc.unsqueeze(0).expand(K, -1)                  # (K, D)
        gf_rep = global_feats_4.unsqueeze(0).expand(K, -1)      # (K, 4)
        b = self.bucket_emb.view(1, -1).expand(K, -1)           # (K, emb)

        edge_full = torch.cat([uv, gc_rep, edge_feats_7, gf_rep, b], dim=1)  # (K, 2D + D + edge_feat_dim + 4 + emb)

        logits = self.edge_mlp(edge_full).squeeze(-1)   # (K,)
        q_pred = self.q_mlp(edge_full).squeeze(-1)      # (K,)
        return logits, value, q_pred

# -----------------------------
# Policy/Value Net (hard isolation across buckets)
# -----------------------------

class PolicyValueNet(nn.Module):
    def __init__(self, node_in, hid, heads, layers, edge_hidden, value_hidden, buckets, bucket_emb, device='cpu'):
        super().__init__()
        self.buckets = nn.ModuleList([
            BucketNet(node_in, hid, heads, layers, edge_hidden, value_hidden, bucket_emb)
            for _ in range(buckets)
        ])

    def forward(self, node_feats, adj_dense, cand_edges, edge_feats_7, bucket_id, global_feats_4):
        # Clamp for safety if args/model mismatch slips through
        if isinstance(bucket_id, torch.Tensor):
            bid = int(bucket_id.item())
        else:
            bid = int(bucket_id)
        bid = max(0, min(bid, len(self.buckets) - 1))
        return self.buckets[bid](node_feats, adj_dense, cand_edges, edge_feats_7, global_feats_4)

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

class PPOAgent:
    def __init__(self, net: PolicyValueNet, cfg: PPOConfig, device='cpu'):
        self.net = net
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
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
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

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
                old_values_list=[]; q_losses=[]
                for (obs_t, a, olp, v, gae, R) in batch:
                    (node_feats, adj_dense, cand_edges, edge_feats, b_id, global_feats, temp, *rest) = obs_t
                    q_aux = rest[0] if len(rest) > 0 else None
                    logits, value, q_pred = self.net(node_feats, adj_dense, cand_edges, edge_feats, b_id, global_feats)
                    dist = Categorical(logits=logits / temp)
                    new_logps.append(dist.log_prob(a))
                    entropies.append(dist.entropy())
                    values.append(value)
                    oldlog.append(olp)
                    actidx.append(a)
                    advantages.append(gae)
                    returns.append(R)
                    old_values_list.append(v)

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

                ratio = torch.exp(new_logps - oldlog)
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
                loss = pol_loss + cfg.vf_coef*val_loss + ent_loss + cfg.aux_q_coef * q_loss_total

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
                    # accumulate logs and break early from this epoch
                    losses['kl'] += float(approx_kl.detach().cpu())
                    losses['clipfrac'] += float(((torch.abs(ratio-1.0) > self.cfg.clip).float().mean()).detach().cpu())
                    # Skip the normal update for this batch
                    continue

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

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
        self.clear_buffer()
        return losses

    # Save/load with arch metadata
    def save(self, path:str, arch:dict):
        obj = {
            'arch': arch,
            'state_dict': self.net.state_dict(),
            'opt': self.opt.state_dict(),
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

def run_episode(env: GraphBuildEnv, net: PolicyValueNet, agent: PPOAgent, args, train=True, ep: int = 0):
    # Greedy ER baseline for terminal reward (for logging/optionally final assignment)
    adj0 = env.adj.copy()
    base_lam2 = er_greedy_baseline_lambda2(env.n, adj0, env.m, backend=env.backend, device=env.device)

    # --- USE MODEL'S BUCKET COUNT, not CLI ---
    try:
        model_bucket_count = len(net.buckets)
    except Exception:
        model_bucket_count = int(getattr(args, 'dens_buckets', 1))
    model_bucket_count = max(1, int(model_bucket_count))
    b_id = bucket_id_from_density(env.n, env.m, model_bucket_count)

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
        # Candidate ER Top-K
        iu, iv, ers = er_topk(env.n, env.adj, k=args.topk, frac_cap=args.topk_frac)
        if len(iu) == 0:
            _ = env._state(refresh=True)
            break

        # Current (no refresh)
        st = env._state(refresh=False)
        node_feats = torch.tensor(st['node_feats'], dtype=torch.float32, device=agent.device)
        adj_dense = torch.tensor(env.adj, dtype=torch.float32, device=agent.device)
        cand_edges = torch.tensor(np.stack([iu,iv], axis=1), dtype=torch.long, device=agent.device)

        # v8-style edge feats
        z = st['z']                      # complex (n,)
        degn = st['deg_norm']            # (n,)
        progress = float(st['progress'])
        dens_target = float(st['dens_target'])
        lam2_now = float(st['lam2'])

        u = iu; v = iv
        dz = np.abs(z[u] - z[v]).astype(np.float32)
        du = degn[u].astype(np.float32)
        dv = degn[v].astype(np.float32)
        hs = ers.astype(np.float32)
        K = len(u)
        edge_feats_np = np.stack([dz,du,dv,np.full(K, progress, dtype=np.float32),np.full(K, dens_target, dtype=np.float32),np.full(K, lam2_now, dtype=np.float32),hs], axis=1)

        edge_feats = torch.tensor(edge_feats_np, dtype=torch.float32, device=agent.device)

        # Global features for value head
        global_feats = torch.tensor([math.log1p(env.n), dens_target, progress, lam2_now],
                                    dtype=torch.float32, device=agent.device)

        logits, value, q_pred = net(node_feats, adj_dense, cand_edges, edge_feats, b_id, global_feats)
        dist = Categorical(logits=logits / (args.train_temperature if train else 1.0))
        a_idx = dist.sample() if train else torch.argmax(dist.logits)

        # --- per-step diagnostics ---
        with torch.no_grad():
            ent_step = float(dist.entropy().detach().cpu().item())
            diag_acc['entropy_sum'] += ent_step

            logits_np = logits.detach().cpu().numpy()
            # Pearson proxy for monotonicity vs ER
            if logits_np.size > 1:
                c = np.corrcoef(logits_np, ers)[0, 1]
                if np.isfinite(c):
                    diag_acc['corr_sum'] += float(c)
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
                du_np = np.asarray(du, dtype=np.float32)
                dv_np = np.asarray(dv, dtype=np.float32)

                rec = {
                    'ep': int(ep),
                    'step': int(total_steps+1),
                    'n': int(env.n),
                    'm': int(env.m),
                    'dens': float(normalized_density(env.n, env.m)),
                    'bucket_id': int(b_id),
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
            agent.store(
                (
                    node_feats.detach(),
                    adj_dense.detach(),
                    cand_edges.detach(),
                    edge_feats.detach(),          # store 7-dim edge feats
                    b_id,
                    global_feats.detach(),
                    torch.tensor(args.train_temperature, device=agent.device),
                    q_aux if getattr(args, 'rollout_reward', False) else None
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
    p.add_argument('--init', type=str, default='path')

    # Heuristic pruning
    p.add_argument('--heuristic', type=str, default='er')
    p.add_argument('--topk', type=int, default=32)
    p.add_argument('--topk_frac', type=float, default=0.2)

    # Spectral
    p.add_argument('--spectral_backend', type=str, default='scipy', choices=['scipy','torch'])
    p.add_argument('--fast_spectral', action='store_true')
    p.add_argument('--spectral_refresh_k', type=int, default=6)

    # Model arch
    p.add_argument('--gat_hidden', type=int, default=64)
    p.add_argument('--gat_heads', type=int, default=6)
    p.add_argument('--gat_layers', type=int, default=3)
    p.add_argument('--edge_mlp_hidden', type=int, default=256)
    p.add_argument('--value_mlp_hidden', type=int, default=128)
    p.add_argument('--dens_buckets', type=int, default=20)
    p.add_argument('--bucket_emb_dim', type=int, default=8)

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

    # Runtime
    p.add_argument('--episodes', type=int, default=2000)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')

    # IO
    p.add_argument('--save_model', type=str, default='v9.pt')
    p.add_argument('--load_model', type=str, default='')
    p.add_argument('--save_every', type=int, default=200)
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
        'buckets': args.dens_buckets,
        'bucket_emb': args.bucket_emb_dim
    }

    net = PolicyValueNet(arch['node_in'], arch['hid'], arch['heads'], arch['layers'],
                         arch['edge_mlp_hidden'], arch['value_mlp_hidden'],
                         arch['buckets'], arch['bucket_emb'],
                         device=device.type).to(device)
    ppo_cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        ppo_epochs=args.ppo_epochs, mb_size=args.mb_size, train_temperature=args.train_temperature,
        target_kl=args.target_kl, clip_vf=args.clip_vf, lr_min=args.lr_min, aux_q_coef=args.aux_q_coef
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
                arch['buckets'], arch['bucket_emb'], device=device.type
            ).to(device)
            agent = PPOAgent(net, ppo_cfg, device=device)

        # Load weights/opt
        net.load_state_dict(obj['state_dict'], strict=True)
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

    def sample_task():
        if not args.multi_task:
            return args.n, args.m
        n = random.randint(args.n_min, args.n_max)
        Mmax = n*(n-1)//2
        base = n - 1
        span = max(1, Mmax - base)

        # Map raw density bounds [0..1] to augmented density bounds (tree-normalized) in [0..1]
        def raw_to_aug(d_raw: float) -> float:
            m_raw = d_raw * Mmax
            return float(np.clip((m_raw - base) / span, 0.0, 1.0))

        aug_lo = raw_to_aug(args.dens_min)
        aug_hi = raw_to_aug(args.dens_max)

        # Choose a bucket uniformly within [aug_lo, aug_hi]
        B = args.dens_buckets
        epsfix = np.nextafter(0.0, -1.0)  # keep the upper edge exclusive
        b_lo = max(0, min(B - 1, int(math.floor(aug_lo * B))))
        b_hi = max(0, min(B - 1, int(math.floor((aug_hi * B) + epsfix))))
        if b_hi < b_lo:
            b_hi = b_lo

        b = random.randint(b_lo, b_hi)
        left = b / B
        right = (b + 1) / B
        dens_aug = left + (right - left) * random.random()

        m = base + int(round(dens_aug * span))
        m = max(base, min(Mmax, m))
        return n, m

    if args.inference_only:
        n, m = args.n, args.m
        cfg = EnvConfig(n=n, m=m, init=args.init, spectral_backend=args.spectral_backend, device=device.type, fast_spectral=False, spectral_refresh_k=args.spectral_refresh_k)
        env = GraphBuildEnv(cfg)
        lam2, reward, steps, base_lam2, diag = run_episode(env, net, agent, args, train=False, ep=0)
        print(f"=== Inference-only ===")
        print(f"Heuristic=ER TopK={args.topk}  greedy=True  n={args.n} m={args.m}")
        print(f"λ2 ≈ {lam2:.6f}  base≈{base_lam2:.6f}  reward={reward:+.6f} ({args.reward_mode})  steps={steps}")
        print(f"pick@1={diag['pick_top1_rate']:.2f}  pick@5={diag['pick_top5_rate']:.2f}  avgERrank={diag['avg_er_rank']:.1f}  corr(logit,ER)={diag['logit_er_corr']:.2f}  H={diag['mean_entropy']:.2f}")
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
    t0=time.time()
    for ep in range(1, args.episodes+1):
        n, m = sample_task()
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
            print(
                f"[ep {ep:5d}] n={n:2d} m={m:4d} dens={dens:.2f} λ2={lam2:.3f} base={base_lam2:.3f} reward={reward:+.3f} steps={steps:3d} | upd loss={stats['loss']:.4f} pol={stats['pol']:.4f} val={stats['val']:.4f} q={stats['q']:.4f} ent={stats['ent']:.4f} kl={stats['kl']:.4f} clipfrac={stats['clipfrac']:.2f} lr={agent.opt.param_groups[0]['lr']:.2e} ent_coef={agent.cfg.ent_coef:.3f}"
                + f" pick@1={diag['pick_top1_rate']:.2f} pick@5={diag['pick_top5_rate']:.2f} avgERrank={diag['avg_er_rank']:.1f} corr={diag['logit_er_corr']:.2f} H={diag['mean_entropy']:.2f}"
            )
        else:
            print(
                f"[ep {ep:5d}] n={n:2d} m={m:4d} dens={dens:.2f} λ2={lam2:.3f} base={base_lam2:.3f} reward={reward:+.3f} steps={steps:3d} | no-update lr={agent.opt.param_groups[0]['lr']:.2e} ent_coef={agent.cfg.ent_coef:.3f}"
                + f" pick@1={diag['pick_top1_rate']:.2f} pick@5={diag['pick_top5_rate']:.2f} avgERrank={diag['avg_er_rank']:.1f} corr={diag['logit_er_corr']:.2f} H={diag['mean_entropy']:.2f}"
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
                      "pick_top1","pick_top5","avg_er_rank","corr_logit_er","mean_entropy"]
            if (ep % args.batch_episodes)==0:
                # we just computed stats
                log_row(args.log_csv, header, [
                    ep, n, m, f"{dens:.6f}", f"{lam2:.6f}", f"{base_lam2:.6f}", f"{reward:.6f}", steps,
                    1, f"{stats['loss']:.6f}", f"{stats['pol']:.6f}", f"{stats['val']:.6f}", f"{stats['q']:.6f}",
                    f"{stats['ent']:.6f}", f"{stats['kl']:.6f}", f"{stats['clipfrac']:.6f}",
                    f"{diag['pick_top1_rate']:.6f}", f"{diag['pick_top5_rate']:.6f}",
                    f"{diag['avg_er_rank']:.6f}", f"{diag['logit_er_corr']:.6f}", f"{diag['mean_entropy']:.6f}"
                ])
            else:
                log_row(args.log_csv, header, [
                    ep, n, m, f"{dens:.6f}", f"{lam2:.6f}", f"{base_lam2:.6f}", f"{reward:.6f}", steps,
                    0, "", "", "", "", "", "",
                    f"{diag['pick_top1_rate']:.6f}", f"{diag['pick_top5_rate']:.6f}",
                    f"{diag['avg_er_rank']:.6f}", f"{diag['logit_er_corr']:.6f}", f"{diag['mean_entropy']:.6f}"
                ])

        if args.save_every>0 and (ep % args.save_every)==0:
            path_ep = episodic_path(args.save_model, ep)
            agent.save(path_ep, arch)
            print(f"[save] -> {path_ep}")

    agent.save(args.save_model, arch)
    print(f"[done] saved -> {args.save_model} | elapsed={time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()



"""
python v9.py --train \
  --episodes 500 \
  --multi_task --n_min 24 --n_max 40 --dens_min 0.25 --dens_max 0.299 \
  --init path --seed 0 \
  --heuristic er --topk 32 --topk_frac 0.2 \
  --train_temperature 0.8 \
  --gat_hidden 128 --gat_heads 6 --gat_layers 3 \
  --edge_mlp_hidden 256 --value_mlp_hidden 128 \
  --dens_buckets 20 \
  --lr 2e-4 --gamma 0.99 --lam 0.95 --clip 0.15 --ent_coef 0.03 --vf_coef 0.6 \
  --ppo_epochs 6 --mb_size 2048 --batch_episodes 4 \
  --threads 8 --device cpu \
  --spectral_backend scipy --spectral_refresh_k 6 \
  --save_model v9_bucket_025_030.pt --save_every 4 --reward_mode logratio
"""




