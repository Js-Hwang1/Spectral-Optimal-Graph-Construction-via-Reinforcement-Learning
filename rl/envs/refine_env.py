"""
REFINE Environment for G(n,m) graph optimization.

Strict (N,M) compliance: the graph MUST return to exactly M edges after every step.

Two-phase iterative refinement:
  1. ADD phase:    node-centric, each node i selects a non-neighbor -> N' unique edges added
  2. REMOVE phase: remove exactly N' specific edges (chosen from edge-level distribution)

The REMOVE phase receives explicit edge indices (K, 2) to remove, guaranteeing
n_removed == n_added. If bridge constraints reduce the safe removal budget below
n_added, cap_to_safe_budget() rolls back excess adds to maintain M.

Bridge masking in REMOVE prevents disconnection (Tarjan's algorithm, O(N+M)).
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from utils.spectral import (exact_lambda2_np, is_connected, find_bridges,
                            lanczos_fiedler_ext, lanczos_fiedler_ext_k, rr_update)
from envs.gnm_env import (build_random_tree_initial, build_ring_random_initial,
                          build_fv_initial,
                          compute_node_features, algebraic_connectivity)

K_STEPS = 20
RR_K = 8  # Rayleigh-Ritz subspace dimension (track v₂..v₉)
DISCONNECT_PENALTY = -10.0


@dataclass
class RefineEnvConfig:
    """Configuration for REFINE environment."""
    min_n: int = 8
    max_n: int = 16
    k_steps: int = K_STEPS
    delta: int = 2
    init_method: str = "ring"  # "ring" (ring+random) or "fv" (FV greedy)


class RefineEnv:
    """
    REFINE environment with strict (N,M) edge count preservation.

    Episode flow:
        1. reset(n, m) -> random spanning tree + random edges
        2. For k in 0..K-1:
            a. add_phase(targets)         -> N' unique edges added
            b. cap_to_safe_budget()       -> rollback excess if safe budget < N'
            c. remove_phase(edge_indices) -> remove exactly effective_n edges
            d. compute_reward()           -> asserts m_current == m, returns delta-lambda2
        3. Episode ends after K steps
    """

    def __init__(
        self,
        config: RefineEnvConfig,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.base_seed = seed
        self.episode_count = 0
        self.inference_mode = False

        # State
        self.adj: Optional[np.ndarray] = None
        self.n: int = 0
        self.m: int = 0
        self.m_current: int = 0
        self.step: int = 0
        self.lambda2: float = 0.0
        self.initial_lambda2: float = 0.0
        self._edges_added_this_step: int = 0
        self._edges_removed_this_step: int = 0
        self._added_edges: List[Tuple[int, int]] = []
        self.v2: Optional[np.ndarray] = None  # warm Lanczos state
        self.V_rr: Optional[np.ndarray] = None  # (N, RR_K) RR subspace
        self.lams_rr: Optional[np.ndarray] = None  # (RR_K,) RR eigenvalues

    def reset(self, n: Optional[int] = None, m: Optional[int] = None) -> dict:
        """Reset environment with new (n, m)."""
        self.episode_count += 1
        seed = self.base_seed + self.episode_count if self.base_seed else None
        self.rng = np.random.RandomState(seed)

        if n is None:
            n = self.rng.randint(self.config.min_n, self.config.max_n + 1)
        if m is None:
            min_m = n - 1
            max_m = n * (n - 1) // 2
            m = self.rng.randint(min_m, max_m + 1)

        min_m = n - 1
        max_m = n * (n - 1) // 2
        m = max(min_m, min(m, max_m))

        self.n = n
        self.m = m
        self.m_current = m
        self.step = 0
        self._edges_added_this_step = 0
        self._edges_removed_this_step = 0
        self._added_edges = []
        self.v2 = None  # cold start on first Lanczos call
        self.V_rr = None
        self.lams_rr = None

        # Build initial graph
        if self.config.init_method == "fv":
            self.adj = build_fv_initial(n, m)
        else:
            self.adj = build_ring_random_initial(n, m, self.rng)
        self.degrees = self.adj.sum(axis=1)

        if self.inference_mode:
            self.lambda2 = 0.0
            self.initial_lambda2 = 0.0
        else:
            self.lambda2 = exact_lambda2_np(self.adj)
            self.initial_lambda2 = self.lambda2

        return {
            'n': self.n,
            'm': self.m,
            'initial_lambda2': self.initial_lambda2,
        }

    def get_node_features(self) -> np.ndarray:
        """Compute (N, 3) node features: [degree_norm, 2-hop, 3-hop]."""
        return compute_node_features(self.adj, self.degrees, self.n)

    def get_spectral_info(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute v₂, v₃, λ₂, λ₃ via single Lanczos run. Updates v2 for warm-starting.

        Returns:
            v2: (N,) approximate Fiedler vector
            v3: (N,) approximate third eigenvector
            lam2: Ritz value approximating λ₂
            lam3: Ritz value approximating λ₃
        """
        v2, v3, lam2, lam3 = lanczos_fiedler_ext(self.adj, k=15, v_init=self.v2)
        self.v2 = v2
        return v2, v3, lam2, lam3

    def init_rr(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Run Lanczos and initialize k-dim Rayleigh-Ritz subspace.

        Returns v₂, v₃, λ₂, λ₃ (same interface as get_spectral_info) but also
        sets self.V_rr (N, RR_K) and self.lams_rr (RR_K,) for tracking.
        """
        V, lams = lanczos_fiedler_ext_k(
            self.adj, k=15, v_init=self.v2, n_eig=RR_K
        )
        self.V_rr = V
        self.lams_rr = lams
        self.v2 = V[:, 0].copy()  # warm start for next Lanczos
        v2 = V[:, 0]
        v3 = V[:, 1] if V.shape[1] > 1 else np.zeros(self.n)
        lam2 = float(lams[0])
        lam3 = float(lams[1]) if len(lams) > 1 else lam2
        return v2, v3, lam2, lam3

    def rr_update_add(self, u: int, v: int):
        """Update RR subspace after adding edge (u, v). O(Nk²)."""
        if self.V_rr is not None:
            self.V_rr, self.lams_rr = rr_update(
                self.V_rr, self.lams_rr, u, v, sign=+1.0
            )
            self.v2 = self.V_rr[:, 0].copy()

    def rr_update_remove(self, u: int, v: int):
        """Update RR subspace after removing edge (u, v). O(Nk²)."""
        if self.V_rr is not None:
            self.V_rr, self.lams_rr = rr_update(
                self.V_rr, self.lams_rr, u, v, sign=-1.0
            )
            self.v2 = self.V_rr[:, 0].copy()

    def get_rr_spectral_info(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Get v₂, v₃, λ₂, λ₃ from current RR state (no Lanczos call).
        Must call init_rr() first.
        """
        assert self.V_rr is not None, "Call init_rr() before get_rr_spectral_info()"
        v2 = self.V_rr[:, 0]
        v3 = self.V_rr[:, 1] if self.V_rr.shape[1] > 1 else np.zeros(self.n)
        lam2 = float(self.lams_rr[0])
        lam3 = float(self.lams_rr[1]) if len(self.lams_rr) > 1 else lam2
        return v2, v3, lam2, lam3

    def get_row_features_rr(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (N, 10) edge features for row i using current RR state. O(N).

        Used for autoregressive ADD: score row i, add edges, RR-update, score row i+1.

        Returns:
            row_feat: (N, 10) edge features for pairs (i, 0), (i, 1), ..., (i, N-1)
            graph_feat: (9,) graph-level features
        """
        v2, v3, lam2, lam3 = self.get_rr_spectral_info()
        v4 = self.V_rr[:, 2] if self.V_rr.shape[1] > 2 else np.zeros(self.n)
        v5 = self.V_rr[:, 3] if self.V_rr.shape[1] > 3 else np.zeros(self.n)
        lam4 = float(self.lams_rr[2]) if len(self.lams_rr) > 2 else lam3
        lam5 = float(self.lams_rr[3]) if len(self.lams_rr) > 3 else lam4

        n = self.n
        nm1 = max(n - 1, 1)
        deg_norm = self.degrees / nm1
        step_frac = self.step / max(self.config.k_steps, 1)
        gap_23 = max(lam3 - lam2, 0.0) / max(n, 1)
        gap_34 = max(lam4 - lam3, 0.0) / max(n, 1)
        gap_45 = max(lam5 - lam4, 0.0) / max(n, 1)

        row_feat = np.zeros((n, 10), dtype=np.float32)
        row_feat[:, 0] = n * (v2[i] - v2[:]) ** 2
        row_feat[:, 1] = n * (v3[i] - v3[:]) ** 2
        row_feat[:, 2] = n * (v4[i] - v4[:]) ** 2
        row_feat[:, 3] = n * (v5[i] - v5[:]) ** 2
        row_feat[:, 4] = deg_norm[i]
        row_feat[:, 5] = deg_norm[:]
        row_feat[:, 6] = gap_23
        row_feat[:, 7] = gap_34
        row_feat[:, 8] = gap_45
        row_feat[:, 9] = step_frac

        graph_feat = np.array([
            deg_norm.mean(),
            lam2 / max(n, 1),
            lam3 / max(n, 1),
            lam4 / max(n, 1),
            lam5 / max(n, 1),
            gap_23,
            gap_34,
            gap_45,
            step_frac,
        ], dtype=np.float32)

        return row_feat, graph_feat

    def begin_add_phase(self):
        """Reset ADD phase counters. Call before autoregressive add_single_edge()."""
        self._edges_added_this_step = 0
        self._edges_removed_this_step = 0
        self._added_edges = []

    def add_single_edge(self, i: int, j: int) -> bool:
        """
        Add a single edge (i,j), update degrees and RR subspace. O(Nk²).

        For autoregressive ADD: call after scoring row i, before scoring row i+1.

        Returns:
            True if edge was actually added, False if skipped (self-loop or exists).
        """
        if i == j or self.adj[i, j] > 0:
            return False
        self.adj[i, j] = 1.0
        self.adj[j, i] = 1.0
        self.degrees[i] += 1
        self.degrees[j] += 1
        edge = (min(i, j), max(i, j))
        self._added_edges.append(edge)
        self._edges_added_this_step += 1
        self.m_current += 1
        self.rr_update_add(i, j)
        return True

    # === REFINE+ (rich features) ===

    def compute_full_eigendecomposition(self):
        """
        Full eigendecomposition of the Laplacian. O(N³).
        Sets self.eigvals (N,) and self.eigvecs (N, N).
        Call once per K-step for rich features.
        """
        L = np.diag(self.degrees.astype(float)) - self.adj.astype(float)
        self.eigvals, self.eigvecs = np.linalg.eigh(L)

    def compute_effective_resistance(self):
        """
        Compute (N, N) effective resistance matrix from eigendecomposition. O(N²).
        Must call compute_full_eigendecomposition() first.

        R(i,j) = L⁺[i,i] + L⁺[j,j] - 2*L⁺[i,j]
        where L⁺ = V[:,1:] @ diag(1/λ[1:]) @ V[:,1:]^T

        We only need diag(L⁺) and L⁺ itself for R. But storing full L⁺ is O(N²).
        Optimization: R[i,j] = sum_k>=1 (v_k[i] - v_k[j])² / λ_k
        Compute via broadcasting on the top eigenvectors.
        """
        assert self.eigvals is not None, "Call compute_full_eigendecomposition() first"
        n = self.n
        # Skip λ₁≈0, use λ₂..λ_n
        lams = self.eigvals[1:]  # (N-1,)
        vecs = self.eigvecs[:, 1:]  # (N, N-1)

        # Avoid division by zero for near-zero eigenvalues
        inv_lams = np.where(lams > 1e-10, 1.0 / lams, 0.0)  # (N-1,)

        # L⁺ diagonal: diag_L_pinv[i] = sum_k v_k[i]² / λ_k
        diag_L_pinv = (vecs ** 2 @ inv_lams)  # (N,)

        # L⁺[i,j] = sum_k v_k[i]*v_k[j] / λ_k
        # R[i,j] = diag[i] + diag[j] - 2*L⁺[i,j]
        # = diag[i] + diag[j] - 2 * sum_k v_k[i]*v_k[j]/λ_k
        # = sum_k (v_k[i] - v_k[j])² / λ_k

        # Efficient: weighted_vecs = vecs * sqrt(inv_lams), then
        # R[i,j] = ||weighted_vecs[i] - weighted_vecs[j]||²
        weighted = vecs * np.sqrt(inv_lams)[None, :]  # (N, N-1)
        # R = ||w_i||² + ||w_j||² - 2 w_i · w_j
        norms_sq = (weighted ** 2).sum(axis=1)  # (N,)
        self.eff_resistance = (norms_sq[:, None] + norms_sq[None, :]
                               - 2.0 * weighted @ weighted.T)  # (N, N)

        # Mean effective resistance for normalization
        # mean_R = 2 * trace(L⁺) / n = 2 * sum(1/λ_k) / n
        self.eff_resistance_mean = max(2.0 * inv_lams.sum() / n, 1e-10)

    def compute_common_neighbors(self):
        """
        Compute (N, N) common neighbors matrix: cn[i,j] = |N(i) ∩ N(j)| = (A²)[i,j].
        O(N²M) via sparse-aware multiply, ≤ O(N³) for dense graphs.
        """
        # For small/medium n, dense matmul is fine
        self.common_neighbors = (self.adj @ self.adj).astype(np.float32)

    def update_common_neighbors_add(self, u: int, v: int):
        """Incrementally update common neighbors after adding edge (u,v). O(N)."""
        if self.common_neighbors is None:
            return
        # Adding edge (u,v): for all w, if w is neighbor of v, then cn(u,w) += 1
        # and if w is neighbor of u, then cn(v,w) += 1
        neighbors_v = np.where(self.adj[v] > 0)[0]
        self.common_neighbors[u, neighbors_v] += 1
        self.common_neighbors[neighbors_v, u] += 1
        neighbors_u = np.where(self.adj[u] > 0)[0]
        self.common_neighbors[v, neighbors_u] += 1
        self.common_neighbors[neighbors_u, v] += 1
        # Also: cn(u,v) and cn(v,u) change (u and v are now neighbors of each other)
        # But A² counts paths of length 2, not direct edges, so this is already handled

    def update_common_neighbors_remove(self, u: int, v: int):
        """Incrementally update common neighbors after removing edge (u,v). O(N)."""
        if self.common_neighbors is None:
            return
        neighbors_v = np.where(self.adj[v] > 0)[0]
        self.common_neighbors[u, neighbors_v] -= 1
        self.common_neighbors[neighbors_v, u] -= 1
        neighbors_u = np.where(self.adj[u] > 0)[0]
        self.common_neighbors[v, neighbors_u] -= 1
        self.common_neighbors[neighbors_u, v] -= 1

    def init_rich_features(self):
        """
        Initialize rich features for REFINE+. Call once per K-step.
        Runs full eigendecomposition O(N³), computes R and cn matrices.
        Also initializes RR subspace from exact eigenvectors.
        """
        self.compute_full_eigendecomposition()
        self.compute_effective_resistance()
        self.compute_common_neighbors()

        # Initialize RR subspace from exact eigenvectors (better than Lanczos)
        n_eig = min(RR_K, self.n - 1)
        self.V_rr = self.eigvecs[:, 1:1+n_eig].copy()
        self.lams_rr = self.eigvals[1:1+n_eig].copy()
        self.v2 = self.V_rr[:, 0].copy()

    def get_edge_level_features_rich(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 10-dim edge features for REFINE+. O(N²).
        Must call init_rich_features() first for R, cn, eigenvalues.
        Uses RR-tracked v₂, v₃ for spectral gaps (fresh between swaps).

        Edge features (N, N, 10):
            0: n * |v₂ᵢ - v₂ⱼ|²       — Fiedler gap (what FV uses)
            1: n * |v₃ᵢ - v₃ⱼ|²       — secondary bottleneck
            2: R(i,j) / R_mean         — effective resistance (what ER uses)
            3: cn(i,j) / (n-2)         — common neighbors fraction
            4: degᵢ / (n-1)            — source degree
            5: degⱼ / (n-1)            — target degree
            6: (λ₃ - λ₂) / n           — spectral gap stability
            7: (λ₄ - λ₂) / n           — spectral width
            8: step / K                — budget awareness
            9: 0.0                      — swap progress (set by caller)

        Graph features (7,):
            0: mean_degree / (n-1)
            1: λ₂ / n
            2: λ₃ / n
            3: λ₄ / n
            4: (λ₃ - λ₂) / n
            5: R_mean * n
            6: step / K
        """
        v2, v3, lam2, lam3 = self.get_rr_spectral_info()
        n = self.n
        nm1 = max(n - 1, 1)
        deg_norm = self.degrees / nm1
        step_frac = self.step / max(self.config.k_steps, 1)
        spectral_gap = max(lam3 - lam2, 0.0) / max(n, 1)

        # λ₄ from RR if available, else from eigendecomposition
        if self.lams_rr is not None and len(self.lams_rr) > 2:
            lam4 = float(self.lams_rr[2])
        elif self.eigvals is not None and len(self.eigvals) > 3:
            lam4 = float(self.eigvals[3])
        else:
            lam4 = lam3
        spectral_width = max(lam4 - lam2, 0.0) / max(n, 1)

        edge_feat = np.zeros((n, n, 10), dtype=np.float32)
        edge_feat[:, :, 0] = n * (v2[:, None] - v2[None, :]) ** 2
        edge_feat[:, :, 1] = n * (v3[:, None] - v3[None, :]) ** 2
        edge_feat[:, :, 2] = self.eff_resistance / self.eff_resistance_mean
        edge_feat[:, :, 3] = self.common_neighbors / max(n - 2, 1)
        edge_feat[:, :, 4] = deg_norm[:, None]
        edge_feat[:, :, 5] = deg_norm[None, :]
        edge_feat[:, :, 6] = spectral_gap
        edge_feat[:, :, 7] = spectral_width
        edge_feat[:, :, 8] = step_frac
        edge_feat[:, :, 9] = 0.0  # swap progress, set by training loop

        graph_feat = np.array([
            deg_norm.mean(),
            lam2 / max(n, 1),
            lam3 / max(n, 1),
            lam4 / max(n, 1),
            spectral_gap,
            self.eff_resistance_mean * n,
            step_frac,
        ], dtype=np.float32)

        return edge_feat, graph_feat

    def get_edge_level_features_rr(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute edge-level features using RR-tracked Fiedler vector.
        4-dim edge features: Fiedler gap, degrees, step fraction.
        v3/v4/v5 and spectral gap scalars removed (PoC: noise, not signal).
        """
        v2, _, lam2, _ = self.get_rr_spectral_info()

        n = self.n
        nm1 = max(n - 1, 1)
        deg_norm = self.degrees / nm1
        step_frac = self.step / max(self.config.k_steps, 1)

        edge_feat = np.zeros((n, n, 4), dtype=np.float32)
        edge_feat[:, :, 0] = n * (v2[:, None] - v2[None, :]) ** 2
        edge_feat[:, :, 1] = deg_norm[:, None]
        edge_feat[:, :, 2] = deg_norm[None, :]
        edge_feat[:, :, 3] = step_frac

        graph_feat = np.array([
            deg_norm.mean(),
            lam2 / max(n, 1),
            step_frac,
        ], dtype=np.float32)

        return edge_feat, graph_feat

    def get_edge_level_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute edge-level spectral features for MLP scoring.

        Edge features (N, N, 6):
            0: n * |v₂ᵢ - v₂ⱼ|²  — normalized Fiedler gap
            1: n * |v₃ᵢ - v₃ⱼ|²  — normalized v₃ gap
            2: degᵢ / (n-1)       — source degree
            3: degⱼ / (n-1)       — target degree
            4: (λ₃ - λ₂) / n     — normalized spectral gap
            5: step / K           — step fraction

        Graph features (5,):
            0: mean_degree / (n-1)
            1: λ₂ / n
            2: λ₃ / n
            3: (λ₃ - λ₂) / n
            4: step / K

        All features are O(1) regardless of graph size (for generalization).

        Returns:
            edge_features: (N, N, 6) float32
            graph_features: (5,) float32
        """
        v2, v3, lam2, lam3 = self.get_spectral_info()
        n = self.n
        nm1 = max(n - 1, 1)
        deg_norm = self.degrees / nm1
        step_frac = self.step / max(self.config.k_steps, 1)
        spectral_gap_norm = max(lam3 - lam2, 0.0) / max(n, 1)

        edge_feat = np.zeros((n, n, 6), dtype=np.float32)
        edge_feat[:, :, 0] = n * (v2[:, None] - v2[None, :]) ** 2
        edge_feat[:, :, 1] = n * (v3[:, None] - v3[None, :]) ** 2
        edge_feat[:, :, 2] = deg_norm[:, None]   # broadcast (N,1) -> (N,N)
        edge_feat[:, :, 3] = deg_norm[None, :]   # broadcast (1,N) -> (N,N)
        edge_feat[:, :, 4] = spectral_gap_norm
        edge_feat[:, :, 5] = step_frac

        graph_feat = np.array([
            deg_norm.mean(),
            lam2 / max(n, 1),
            lam3 / max(n, 1),
            spectral_gap_norm,
            step_frac,
        ], dtype=np.float32)

        return edge_feat, graph_feat

    def get_bridge_mask(self) -> np.ndarray:
        """
        Compute (N, N) bridge mask. True for bridge edges.
        Uses Tarjan's algorithm O(N+M).
        """
        bridges = find_bridges(self.adj)
        mask = np.zeros((self.n, self.n), dtype=bool)
        for u, v in bridges:
            mask[u, v] = True
            mask[v, u] = True
        return mask

    def add_phase(self, targets: np.ndarray) -> int:
        """
        ADD phase: each node i selects non-neighbor targets to connect to.

        Supports two shapes:
          - (N,):      each node picks 1 target (delta=1 legacy)
          - (N, delta): each node picks up to delta targets

        Symmetry: canonical edge (min, max). If i->j and j->i both proposed,
        only one undirected edge added. Processing in node order.

        Stores added edges internally for potential rollback via cap_to_safe_budget().

        Args:
            targets: (N,) or (N, delta) array of destination nodes

        Returns:
            n_added: number of unique undirected edges added
        """
        added_edges_set = set()
        self._added_edges = []
        n_added = 0

        if targets.ndim == 1:
            # Legacy: (N,) — each node picks 1 target
            for i in range(self.n):
                j = int(targets[i])
                if i == j:
                    continue
                edge = (min(i, j), max(i, j))
                if edge in added_edges_set:
                    continue
                if self.adj[i, j] == 0:
                    self.adj[i, j] = 1.0
                    self.adj[j, i] = 1.0
                    self.degrees[i] += 1
                    self.degrees[j] += 1
                    added_edges_set.add(edge)
                    self._added_edges.append(edge)
                    n_added += 1
                    self.rr_update_add(i, j)
        else:
            # Delta mode: (N, delta) — each node picks up to delta targets
            delta = targets.shape[1]
            for i in range(self.n):
                for d in range(delta):
                    j = int(targets[i, d])
                    if i == j:
                        continue
                    edge = (min(i, j), max(i, j))
                    if edge in added_edges_set:
                        continue
                    if self.adj[i, j] == 0:
                        self.adj[i, j] = 1.0
                        self.adj[j, i] = 1.0
                        self.degrees[i] += 1
                        self.degrees[j] += 1
                        added_edges_set.add(edge)
                        self._added_edges.append(edge)
                        n_added += 1
                        self.rr_update_add(i, j)

        self._edges_added_this_step = n_added
        self.m_current += n_added
        return n_added

    def cap_to_safe_budget(self) -> int:
        """
        After ADD, check bridge constraints and rollback excess adds if needed.

        Counts non-bridge edges on the post-ADD graph. If safe_removable < n_added,
        rolls back the last (n_added - safe_removable) added edges so that the
        REMOVE phase can safely remove exactly effective_n edges.

        Must be called AFTER add_phase() and BEFORE remove_phase().

        Returns:
            effective_n: number of edges that should be removed (== final n_added)
        """
        n_added = self._edges_added_this_step
        if n_added == 0:
            return 0

        bridges = find_bridges(self.adj)
        bridge_set = set()
        for u, v in bridges:
            bridge_set.add((min(u, v), max(u, v)))

        # Count non-bridge existing edges (upper triangle)
        safe_count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj[i, j] > 0 and (i, j) not in bridge_set:
                    safe_count += 1

        effective_n = min(n_added, safe_count)

        if effective_n < n_added:
            # Rollback excess added edges (from the end of the list)
            excess = n_added - effective_n
            to_undo = self._added_edges[-excess:]
            for i, j in to_undo:
                self.adj[i, j] = 0.0
                self.adj[j, i] = 0.0
                self.degrees[i] -= 1
                self.degrees[j] -= 1
            self._added_edges = self._added_edges[:-excess]
            self._edges_added_this_step = effective_n
            self.m_current -= excess

        return effective_n

    def remove_phase(self, edge_indices: np.ndarray) -> int:
        """
        REMOVE phase: remove edges with sequential connectivity checking.

        Three phases ensure strict (N,M) compliance:
        1. Try removing the policy's preferred edges (skip any that disconnect)
        2. Backfill: if not enough removed, greedily find other safe removals
        3. Fallback: undo excess ADDs if backfill can't fill the gap

        Phase 2 is guaranteed to succeed because the post-ADD graph has cycle
        rank >= n_added (added edges create cycles since the graph was connected).

        Args:
            edge_indices: (K, 2) array where each row is an edge (i, j) to remove.

        Returns:
            n_removed: number of edges actually removed
        """
        if len(edge_indices) == 0:
            self._edges_removed_this_step = 0
            return 0

        n_target = self._edges_added_this_step
        removed_set = set()
        n_removed = 0

        # Phase 1: try the policy's preferred removals
        for k in range(len(edge_indices)):
            if n_removed >= n_target:
                break
            i, j = int(edge_indices[k, 0]), int(edge_indices[k, 1])
            if self.adj[i, j] > 0:
                self.adj[i, j] = 0.0
                self.adj[j, i] = 0.0
                if is_connected(self.adj):
                    self.degrees[i] -= 1
                    self.degrees[j] -= 1
                    removed_set.add((min(i, j), max(i, j)))
                    n_removed += 1
                    self.rr_update_remove(i, j)
                else:
                    self.adj[i, j] = 1.0
                    self.adj[j, i] = 1.0

        # Phase 2: backfill with other safe edges
        if n_removed < n_target:
            for i in range(self.n):
                if n_removed >= n_target:
                    break
                for j in range(i + 1, self.n):
                    if n_removed >= n_target:
                        break
                    if self.adj[i, j] > 0 and (i, j) not in removed_set:
                        self.adj[i, j] = 0.0
                        self.adj[j, i] = 0.0
                        if is_connected(self.adj):
                            self.degrees[i] -= 1
                            self.degrees[j] -= 1
                            removed_set.add((i, j))
                            n_removed += 1
                            self.rr_update_remove(i, j)
                        else:
                            self.adj[i, j] = 1.0
                            self.adj[j, i] = 1.0

        self._edges_removed_this_step = n_removed
        self.m_current -= n_removed

        # Phase 3: undo excess ADDs (shouldn't happen, but safety net)
        shortfall = self._edges_added_this_step - n_removed
        while shortfall > 0 and self._added_edges:
            ai, aj = self._added_edges.pop()
            if self.adj[ai, aj] > 0:
                self.adj[ai, aj] = 0.0
                self.adj[aj, ai] = 0.0
                self.degrees[ai] -= 1
                self.degrees[aj] -= 1
                self._edges_added_this_step -= 1
                self.m_current -= 1
                shortfall -= 1
            else:
                # This added edge was already removed in Phase 1/2
                self._edges_added_this_step -= 1
                shortfall -= 1

        return n_removed

    def compute_reward(self) -> float:
        """
        Compute reward as delta-lambda2. Asserts strict (N,M) compliance.

        Training only — O(N^3).
        """
        # Strict edge count assertion
        assert self.m_current == self.m, \
            f"Edge count violation: m_current={self.m_current} != m={self.m} " \
            f"(added={self._edges_added_this_step}, removed={self._edges_removed_this_step})"

        if self.inference_mode:
            self.step += 1
            return 0.0

        new_lambda2 = exact_lambda2_np(self.adj)

        if new_lambda2 < 1e-6:
            reward = DISCONNECT_PENALTY
        else:
            reward = new_lambda2 - self.lambda2
            self.lambda2 = new_lambda2

        self.step += 1
        return reward

    def get_current_edges(self) -> np.ndarray:
        """Return (M, 2) array of current edges (i < j)."""
        ei, ej = np.where(np.triu(self.adj, 1) > 0)
        return np.stack([ei, ej], axis=1)

    def get_non_edges(self) -> np.ndarray:
        """Return (NE, 2) array of current non-edges (i < j)."""
        mask = np.triu(1.0 - self.adj - np.eye(self.n), 1) > 0
        ni, nj = np.where(mask)
        return np.stack([ni, nj], axis=1)

    @property
    def done(self) -> bool:
        """Episode is done after K steps."""
        return self.step >= self.config.k_steps

    @property
    def final_lambda2(self) -> float:
        """Compute final lambda2 via exact eigensolver."""
        return algebraic_connectivity(self.adj)
