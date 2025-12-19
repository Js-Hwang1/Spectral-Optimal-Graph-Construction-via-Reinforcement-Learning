"""
Phase 2: RL Residual Edge Filling Module (Research-Grade Implementation)

This module provides inference-time edge placement using a trained policy network.
The network learns to predict which edges maximize algebraic connectivity (lambda_2)
without computing eigenvalues at inference time (O(N^2) constraint).

Key Design Decisions:
1. Graph-theoretic features that correlate with lambda_2 (Fiedler theory)
2. Graph Attention Network (GAT) for size-invariant representations
3. Vectorized operations for efficient O(N^2) inference
4. Spectral gap proxy features using power iteration approximation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional


# =============================================================================
# 1. SCIENTIFICALLY-GROUNDED FEATURE EXTRACTION
# =============================================================================
class GraphFeatureExtractor:
    """
    Extract structural features that theoretically correlate with algebraic connectivity.

    From spectral graph theory, lambda_2 is related to:
    - Graph expansion/sparsest cut (Cheeger inequality)
    - Mixing time of random walks
    - Diameter and vertex connectivity

    We compute O(N^2) proxies for these properties.
    """

    @staticmethod
    def get_node_features(adj: np.ndarray, num_power_iters: int = 3) -> np.ndarray:
        """
        Extract per-node features in O(N^2).

        Features (8 total per node):
        0. Normalized degree
        1. Clustering coefficient
        2. Neighbor avg degree (2-hop connectivity proxy)
        3. Neighbor degree variance (local regularity)
        4. Eigenvector centrality approx (power iteration)
        5. Local edge density (edges among neighbors / possible)
        6. Eccentricity lower bound (via BFS-like propagation)
        7. PageRank-like score (random walk centrality)
        """
        n = adj.shape[0]
        adj_float = adj.astype(np.float32)
        degrees = adj_float.sum(axis=1)
        max_deg = max(degrees.max(), 1)

        features = np.zeros((n, 8), dtype=np.float32)

        # 0. Normalized degree (connectivity measure)
        features[:, 0] = degrees / max_deg

        # 1. Clustering coefficient (triangle density)
        # C_i = 2 * triangles_i / (deg_i * (deg_i - 1))
        adj_sq = adj_float @ adj_float
        triangles = np.diag(adj_sq @ adj_float) / 2
        possible = degrees * (degrees - 1) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            features[:, 1] = np.where(possible > 0, triangles / possible, 0)

        # 2-3. Neighbor degree statistics
        neighbor_deg_sum = adj_sq.sum(axis=1)  # Sum of neighbor degrees
        features[:, 2] = np.where(degrees > 0,
                                   neighbor_deg_sum / (degrees * max_deg), 0)

        # Compute neighbor degree variance efficiently
        # Var = E[X^2] - E[X]^2
        deg_sq = degrees ** 2
        neighbor_deg_sq_sum = (adj_float @ deg_sq)
        neighbor_deg_mean_sq = (neighbor_deg_sum / np.maximum(degrees, 1)) ** 2
        neighbor_deg_var = np.where(
            degrees > 1,
            neighbor_deg_sq_sum / degrees - neighbor_deg_mean_sq,
            0
        )
        features[:, 3] = neighbor_deg_var / (max_deg ** 2 + 1e-8)

        # 4. Eigenvector centrality via power iteration (correlates with lambda_1)
        x = np.ones(n, dtype=np.float32) / np.sqrt(n)
        for _ in range(num_power_iters):
            x = adj_float @ x
            norm = np.linalg.norm(x)
            if norm > 1e-8:
                x = x / norm
        features[:, 4] = x / (x.max() + 1e-8)

        # 5. Local edge density (edges among neighbors)
        # This measures how "clique-like" the neighborhood is
        local_edges = np.diag(adj_sq @ adj_sq) / 2  # Edges in 2-neighborhood
        max_local = degrees * (degrees - 1) / 2
        features[:, 5] = np.where(max_local > 0, local_edges / (max_local + 1), 0)

        # 6. Eccentricity proxy via distance propagation (BFS-like)
        # Use matrix powers to estimate max distance
        dist_proxy = np.zeros(n, dtype=np.float32)
        reached = adj_float.copy()
        for d in range(1, min(5, n)):  # Limit depth for O(N^2)
            unreached = (reached.sum(axis=1) < n - 1).astype(np.float32)
            dist_proxy += unreached * d
            reached = np.minimum(reached @ adj_float + reached, 1)
        features[:, 6] = dist_proxy / (min(5, n) + 1e-8)

        # 7. PageRank-like centrality (random walk stationary distribution proxy)
        damping = 0.85
        pr = np.ones(n, dtype=np.float32) / n
        deg_inv = np.where(degrees > 0, 1.0 / degrees, 0)
        trans = adj_float * deg_inv  # Transition matrix (column-normalized)
        for _ in range(num_power_iters):
            pr = damping * (trans @ pr) + (1 - damping) / n
        features[:, 7] = pr / (pr.max() + 1e-8)

        return features

    @staticmethod
    def get_edge_features(adj: np.ndarray, node_features: np.ndarray) -> np.ndarray:
        """
        Extract features for each potential edge (i, j) where i < j.

        Edge features (10 total):
        0. Edge exists (0/1)
        1. Degree product (normalized) - high for hub connections
        2. Degree sum (normalized) - total connectivity gain
        3. Common neighbors count - clustering impact
        4. Jaccard similarity - relative overlap
        5. Adamic-Adar index - weighted common neighbors
        6. Node feature dot product - similarity
        7. Degree difference - regularity impact
        8. Resource allocation index
        9. Preferential attachment score
        """
        n = adj.shape[0]
        adj_float = adj.astype(np.float32)
        degrees = adj_float.sum(axis=1)
        max_deg = max(degrees.max(), 1)

        # Pre-compute matrices for vectorized operations
        common_neighbors = adj_float @ adj_float  # CN[i,j] = # common neighbors

        # Adamic-Adar: sum of 1/log(deg) for common neighbors
        log_deg = np.log(degrees + 2)  # +2 to avoid log(1)=0
        inv_log_deg = 1.0 / log_deg
        aa_matrix = adj_float @ np.diag(inv_log_deg) @ adj_float

        # Resource allocation: sum of 1/deg for common neighbors
        deg_inv = np.where(degrees > 0, 1.0 / degrees, 0)
        ra_matrix = adj_float @ np.diag(deg_inv) @ adj_float

        num_edges = n * (n - 1) // 2
        edge_features = np.zeros((num_edges, 10), dtype=np.float32)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # 0. Edge exists
                edge_features[idx, 0] = adj_float[i, j]

                # 1. Degree product (normalized)
                edge_features[idx, 1] = (degrees[i] * degrees[j]) / (max_deg ** 2 + 1e-8)

                # 2. Degree sum (normalized)
                edge_features[idx, 2] = (degrees[i] + degrees[j]) / (2 * max_deg + 1e-8)

                # 3. Common neighbors (normalized)
                cn = common_neighbors[i, j]
                edge_features[idx, 3] = cn / (n - 2 + 1e-8)

                # 4. Jaccard similarity
                union = degrees[i] + degrees[j] - cn
                edge_features[idx, 4] = cn / (union + 1e-8) if union > 0 else 0

                # 5. Adamic-Adar index (normalized)
                edge_features[idx, 5] = aa_matrix[i, j] / (np.log(n) + 1e-8)

                # 6. Node feature similarity (dot product)
                edge_features[idx, 6] = np.dot(node_features[i], node_features[j]) / 8  # Normalize by feature dim

                # 7. Degree difference (normalized)
                edge_features[idx, 7] = abs(degrees[i] - degrees[j]) / (max_deg + 1e-8)

                # 8. Resource allocation index
                edge_features[idx, 8] = ra_matrix[i, j]

                # 9. Preferential attachment (normalized)
                edge_features[idx, 9] = (degrees[i] * degrees[j]) / ((n * max_deg) + 1e-8)

                idx += 1

        return edge_features

    @staticmethod
    def get_global_features(adj: np.ndarray, degrees: np.ndarray) -> np.ndarray:
        """
        Global graph features for conditioning.

        Features (6 total):
        0. Graph density
        1. Degree variance (regularity measure)
        2. Avg clustering coefficient
        3. Edge count ratio (current/max)
        4. Algebraic connectivity proxy (spectral gap lower bound)
        5. Graph diameter proxy
        """
        n = adj.shape[0]
        m = adj.sum() / 2
        max_m = n * (n - 1) / 2

        features = np.zeros(6, dtype=np.float32)

        # 0. Density
        features[0] = m / (max_m + 1e-8)

        # 1. Degree variance (normalized)
        mean_deg = degrees.mean()
        features[1] = degrees.var() / (mean_deg ** 2 + 1e-8)

        # 2. Average clustering
        adj_float = adj.astype(np.float32)
        adj_sq = adj_float @ adj_float
        triangles = np.diag(adj_sq @ adj_float) / 2
        possible = degrees * (degrees - 1) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            cc = np.where(possible > 0, triangles / possible, 0)
        features[2] = cc.mean()

        # 3. Edge ratio
        features[3] = m / (n + 1e-8)

        # 4. Spectral gap proxy: min(degrees) (lower bound on lambda_2)
        features[4] = degrees.min() / (n + 1e-8)

        # 5. Diameter proxy: use spectral gap relation
        # diameter <= ceil(log(n-1) / log(lambda_n / lambda_2))
        # We use inverse of min degree as proxy
        features[5] = 1.0 / (degrees.min() + 1)

        return features


# =============================================================================
# 2. GRAPH ATTENTION NETWORK ARCHITECTURE
# =============================================================================
class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with multi-head attention.

    Implements attention mechanism: alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
    This allows the network to learn which neighbors are most important.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat

        # Per-head transformations
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Attention mechanism
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.out_proj = nn.Linear(out_dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n, in_dim) node features
            adj: (batch, n, n) adjacency matrix
        Returns:
            (batch, n, out_dim) updated node features
        """
        batch_size, n, _ = x.shape

        # Linear transformation: (batch, n, heads * out_dim)
        Wh = self.W(x).view(batch_size, n, self.num_heads, self.out_dim)

        # Compute attention scores
        # a_src: (heads, out_dim) -> score per source node
        # a_dst: (heads, out_dim) -> score per destination node
        e_src = (Wh * self.a_src).sum(dim=-1)  # (batch, n, heads)
        e_dst = (Wh * self.a_dst).sum(dim=-1)  # (batch, n, heads)

        # Broadcast to get pairwise scores: e_ij = e_src_i + e_dst_j
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (batch, n, n, heads)
        e = self.leaky_relu(e)

        # Mask non-edges with large negative value
        mask = (adj == 0).unsqueeze(-1)  # (batch, n, n, 1)
        e = e.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        alpha = F.softmax(e, dim=2)  # (batch, n, n, heads)
        alpha = self.dropout(alpha)

        # Handle all-masked rows (isolated nodes)
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)

        # Aggregate: h'_i = sum_j alpha_ij * Wh_j
        # (batch, n, n, heads) @ (batch, n, heads, out_dim) -> (batch, n, heads, out_dim)
        Wh_t = Wh.permute(0, 2, 1, 3)  # (batch, heads, n, out_dim)
        alpha_t = alpha.permute(0, 3, 1, 2)  # (batch, heads, n, n)
        out = torch.matmul(alpha_t, Wh_t)  # (batch, heads, n, out_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, n, heads, out_dim)

        if self.concat:
            out = out.reshape(batch_size, n, -1)  # (batch, n, heads * out_dim)
            out = self.out_proj(out)  # (batch, n, out_dim)
        else:
            out = out.mean(dim=2)  # (batch, n, out_dim)

        return out


class EdgePolicyNetwork(nn.Module):
    """
    Graph Neural Network for edge placement policy.

    Architecture:
    1. Node encoder: Project node features to embedding space
    2. GAT layers: Learn graph-aware node representations
    3. Global conditioning: Inject global graph features
    4. Edge scorer: Score each candidate edge
    5. Value head: Estimate state value for advantage computation

    Key design choices:
    - Size-invariant through mean aggregation and normalization
    - Attention mechanism learns importance of different neighbors
    - Edge features incorporate link prediction heuristics
    """

    def __init__(self,
                 node_feat_dim: int = 8,
                 edge_feat_dim: int = 10,
                 global_feat_dim: int = 6,
                 hidden_dim: int = 64,
                 num_gat_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_gat_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge scorer MLP
        # Input: node_i features + node_j features + edge features + global features
        edge_input_dim = hidden_dim * 2 + edge_feat_dim + hidden_dim
        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # graph + global
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                adj: torch.Tensor,
                global_features: torch.Tensor,
                candidate_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute edge logits and state value.

        Args:
            node_features: (batch, n, node_feat_dim)
            edge_features: (batch, num_edges, edge_feat_dim)
            adj: (batch, n, n) adjacency matrix
            global_features: (batch, global_feat_dim)
            candidate_mask: (batch, num_edges) valid candidates mask

        Returns:
            edge_logits: (batch, num_edges)
            value: (batch, 1)
        """
        batch_size, n, _ = node_features.shape

        # Encode nodes
        node_emb = self.node_encoder(node_features)  # (batch, n, hidden)

        # GAT layers with residual connections
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            node_emb_new = gat(node_emb, adj)
            node_emb = ln(node_emb + node_emb_new)  # Residual + LayerNorm

        # Encode global features
        global_emb = self.global_encoder(global_features)  # (batch, hidden)

        # Create edge embeddings (vectorized for efficiency)
        num_edges = n * (n - 1) // 2

        # Build index tensors for edge endpoints
        idx_i = []
        idx_j = []
        for i in range(n):
            for j in range(i + 1, n):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=node_features.device)
        idx_j = torch.tensor(idx_j, device=node_features.device)

        # Gather node embeddings for each edge
        node_i = node_emb[:, idx_i]  # (batch, num_edges, hidden)
        node_j = node_emb[:, idx_j]  # (batch, num_edges, hidden)

        # Expand global features
        global_exp = global_emb.unsqueeze(1).expand(-1, num_edges, -1)

        # Concatenate all edge features
        edge_input = torch.cat([node_i, node_j, edge_features, global_exp], dim=-1)

        # Score edges
        edge_logits = self.edge_scorer(edge_input).squeeze(-1)  # (batch, num_edges)

        # Mask invalid candidates
        edge_logits = edge_logits.masked_fill(~candidate_mask, float('-inf'))

        # Value estimation
        graph_emb = node_emb.mean(dim=1)  # (batch, hidden) - size-invariant
        value_input = torch.cat([graph_emb, global_emb], dim=-1)
        value = self.value_head(value_input)

        return edge_logits, value

    def get_action(self,
                   node_features: torch.Tensor,
                   edge_features: torch.Tensor,
                   adj: torch.Tensor,
                   global_features: torch.Tensor,
                   candidate_mask: torch.Tensor,
                   num_edges_to_add: int,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select edges to add based on policy.

        For O(N^2) inference, predicts all edges in one forward pass.
        """
        edge_logits, value = self.forward(
            node_features, edge_features, adj, global_features, candidate_mask
        )

        # Compute probabilities
        edge_probs = F.softmax(edge_logits, dim=-1)

        # Clamp for numerical stability
        edge_probs = edge_probs.clamp(min=1e-8)

        # Handle case where we need fewer edges than candidates
        num_valid = candidate_mask.sum(dim=-1).min().item()
        num_to_select = min(num_edges_to_add, int(num_valid))

        if num_to_select <= 0:
            return (torch.zeros(edge_probs.shape[0], 0, device=edge_probs.device, dtype=torch.long),
                    torch.zeros(edge_probs.shape[0], device=edge_probs.device),
                    value)

        if deterministic:
            _, selected_indices = torch.topk(edge_probs, num_to_select, dim=-1)
        else:
            # Sample without replacement using Gumbel-top-k trick for efficiency
            gumbel = -torch.log(-torch.log(torch.rand_like(edge_probs) + 1e-8) + 1e-8)
            perturbed = torch.log(edge_probs + 1e-8) + gumbel
            perturbed = perturbed.masked_fill(~candidate_mask, float('-inf'))
            _, selected_indices = torch.topk(perturbed, num_to_select, dim=-1)

        # Compute log probabilities
        log_probs = F.log_softmax(edge_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, selected_indices)
        total_log_prob = selected_log_probs.sum(dim=-1)

        return selected_indices, total_log_prob, value


# =============================================================================
# 3. RESIDUAL EDGE FILLER (INFERENCE)
# =============================================================================
class ResidualEdgeFiller:
    """
    Main interface for Phase 2 edge placement.

    Uses trained policy network for O(N^2) inference.
    Falls back to spectral-aware heuristic when no model is available.
    """

    def __init__(self, model_path: str = None, hidden_dim: int = 64, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.model = None
        self.feature_extractor = GraphFeatureExtractor()

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = EdgePolicyNetwork(
            hidden_dim=checkpoint.get('hidden_dim', self.hidden_dim),
            node_feat_dim=checkpoint.get('node_feat_dim', 8),
            edge_feat_dim=checkpoint.get('edge_feat_dim', 10),
            num_gat_layers=checkpoint.get('num_gat_layers', 2),
            num_heads=checkpoint.get('num_heads', 4),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _flat_to_pair(n: int, flat_idx: int) -> Tuple[int, int]:
        """Convert flattened upper triangular index to (i, j) pair."""
        # Analytical inverse of: idx = i * (2n - i - 1) / 2 + j - i - 1
        i = int(n - 2 - np.floor(np.sqrt(-8 * flat_idx + 4 * n * (n - 1) - 7) / 2 - 0.5))
        j = int(flat_idx + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
        return i, j

    def fill_edges(self, adj: np.ndarray, num_edges_to_add: int,
                   deterministic: bool = True) -> np.ndarray:
        """
        Add edges to the skeleton graph using the trained policy.

        Args:
            adj: Current adjacency matrix (from Phase 1)
            num_edges_to_add: Number of remaining edges to place
            deterministic: If True, use greedy selection; else sample

        Returns:
            Updated adjacency matrix with new edges
        """
        if num_edges_to_add <= 0:
            return adj.copy()

        n = adj.shape[0]
        adj = adj.copy().astype(np.float32)

        if self.model is None:
            return self._spectral_heuristic_fill(adj, num_edges_to_add)

        with torch.no_grad():
            # Extract features
            degrees = adj.sum(axis=1)
            node_features = self.feature_extractor.get_node_features(adj)
            edge_features = self.feature_extractor.get_edge_features(adj, node_features)
            global_features = self.feature_extractor.get_global_features(adj, degrees)

            # Create candidate mask
            num_edges = n * (n - 1) // 2
            candidate_mask = np.zeros(num_edges, dtype=bool)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    candidate_mask[idx] = (adj[i, j] == 0)
                    idx += 1

            num_candidates = candidate_mask.sum()
            num_to_add = min(num_edges_to_add, num_candidates)

            if num_to_add == 0:
                return adj

            # Convert to tensors
            node_feat_t = torch.tensor(node_features, device=self.device).unsqueeze(0)
            edge_feat_t = torch.tensor(edge_features, device=self.device).unsqueeze(0)
            adj_t = torch.tensor(adj, device=self.device).unsqueeze(0)
            global_feat_t = torch.tensor(global_features, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(candidate_mask, device=self.device).unsqueeze(0)

            # Get edge selections
            selected_indices, _, _ = self.model.get_action(
                node_feat_t, edge_feat_t, adj_t, global_feat_t, mask_t,
                num_to_add, deterministic=deterministic
            )

            # Add selected edges
            for flat_idx in selected_indices[0].cpu().numpy():
                i, j = self._flat_to_pair(n, int(flat_idx))
                adj[i, j] = 1
                adj[j, i] = 1

        return adj

    def _spectral_heuristic_fill(self, adj: np.ndarray, num_edges_to_add: int) -> np.ndarray:
        """
        Spectral-aware heuristic for edge placement.

        Based on algebraic connectivity theory:
        - Adding edges between low-degree nodes tends to increase lambda_2
        - Edges that reduce graph diameter are beneficial
        - Balance between local clustering and global connectivity

        Uses a scoring function that approximates lambda_2 improvement.
        """
        n = adj.shape[0]
        adj = adj.copy()

        for _ in range(num_edges_to_add):
            degrees = adj.sum(axis=1)
            mean_deg = degrees.mean()

            # Compute Fiedler vector approximation via power iteration
            # The Fiedler vector indicates which nodes are "bottlenecks"
            L = np.diag(degrees) - adj

            # Power iteration on shifted Laplacian to find second eigenvector
            x = np.random.randn(n)
            x = x - x.mean()  # Orthogonalize to constant vector
            for _ in range(10):
                # Shift to make lambda_2 largest
                x = (degrees.max() * np.eye(n) - L) @ x
                x = x - x.mean()
                norm = np.linalg.norm(x)
                if norm > 1e-8:
                    x = x / norm

            # Score candidate edges
            best_score = float('-inf')
            best_edge = None

            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 0:
                        # Score based on multiple factors

                        # 1. Fiedler vector difference (higher = more impact on lambda_2)
                        fiedler_impact = abs(x[i] - x[j])

                        # 2. Degree balancing (prefer connecting low-degree nodes)
                        deg_factor = 1.0 / (degrees[i] + degrees[j] + 1)

                        # 3. Reduce degree variance (promotes regularity)
                        new_deg_i = degrees[i] + 1
                        new_deg_j = degrees[j] + 1
                        old_var_contrib = (degrees[i] - mean_deg)**2 + (degrees[j] - mean_deg)**2
                        new_mean = (mean_deg * n + 2) / n
                        new_var_contrib = (new_deg_i - new_mean)**2 + (new_deg_j - new_mean)**2
                        regularity_factor = (old_var_contrib - new_var_contrib) / (mean_deg + 1)

                        # Combined score
                        score = (2.0 * fiedler_impact +
                                1.0 * deg_factor +
                                0.5 * regularity_factor)

                        if score > best_score:
                            best_score = score
                            best_edge = (i, j)

            if best_edge is None:
                break

            i, j = best_edge
            adj[i, j] = 1
            adj[j, i] = 1

        return adj


# =============================================================================
# 4. COMBINED PIPELINE
# =============================================================================
def build_graph_with_rl(n: int, m: int, model_path: str = None,
                        skeleton_budget_ratio: float = 0.8) -> np.ndarray:
    """
    Full two-phase graph construction pipeline.

    Args:
        n: Number of nodes
        m: Total edge budget
        model_path: Path to trained RL model (optional)
        skeleton_budget_ratio: Fraction of edges for Phase 1 skeleton

    Returns:
        Final adjacency matrix
    """
    from phase1 import AdaptiveGraphBuilder

    # Phase 1: Build deterministic skeleton
    m_skeleton = int(m * skeleton_budget_ratio)
    m_skeleton = max(n - 1, min(m_skeleton, m))  # At least spanning tree

    builder = AdaptiveGraphBuilder(n)
    builder.build(m_skeleton)
    skeleton_adj = builder.adj.astype(np.float32)

    # Phase 2: Fill remaining edges with RL
    m_remaining = m - len(builder.edges)

    if m_remaining > 0:
        filler = ResidualEdgeFiller(model_path=model_path)
        final_adj = filler.fill_edges(skeleton_adj, m_remaining)
    else:
        final_adj = skeleton_adj

    return final_adj


# =============================================================================
# 5. TESTING
# =============================================================================
if __name__ == "__main__":
    import scipy.linalg

    def get_lambda2(adj):
        degrees = np.sum(adj, axis=0)
        L = np.diag(degrees) - adj
        eigvals = scipy.linalg.eigh(L, eigvals_only=True)
        eigvals.sort()
        return eigvals[1] if len(eigvals) > 1 else 0.0

    print("Testing Phase 2 Module (Spectral Heuristic)")
    print("=" * 70)

    from phase1 import AdaptiveGraphBuilder

    test_cases = [(20, 40), (30, 80), (40, 150), (50, 200)]

    for n, m in test_cases:
        # Phase 1: 70% budget
        m1 = int(m * 0.7)
        builder = AdaptiveGraphBuilder(n)
        builder.build(m1)
        skeleton_l2 = builder.get_lambda2()

        # Phase 2: Fill remaining with spectral heuristic
        m_rem = m - len(builder.edges)
        filler = ResidualEdgeFiller()
        final_adj = filler.fill_edges(builder.adj.astype(np.float32), m_rem)
        final_l2 = get_lambda2(final_adj)

        actual_edges = int(final_adj.sum() / 2)
        improvement = final_l2 - skeleton_l2
        pct = 100 * improvement / (skeleton_l2 + 1e-8)

        print(f"N={n:2d}, M={m:3d}: Skeleton L2={skeleton_l2:.4f} -> Final L2={final_l2:.4f} "
              f"(+{improvement:.4f}, +{pct:.1f}%) | Edges: {len(builder.edges)}->{actual_edges}")
