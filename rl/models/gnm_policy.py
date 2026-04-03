"""
GNN Edge Selection Policy — 3-layer message-passing + edge scoring.

Architecture:
  - input_proj: Linear(NODE_FEAT_DIM → hidden_dim)
  - gnn_layers: 3× GNNLayer(hidden_dim → hidden_dim) with D⁻¹A normalization + residual
  - remove_scorer: MLP(2*hidden_dim + GRAPH_FEAT_DIM → hidden_dim → 1)
  - add_scorer:    MLP(2*hidden_dim + GRAPH_FEAT_DIM → hidden_dim → 1)
  - value_head:    MLP(hidden_dim + GRAPH_FEAT_DIM → hidden_dim → 1)

R is deterministic = min(m//4, M, NE). Policy decides WHICH edges to swap.
GNN propagates structural info across the graph — O(N²d) per layer, O(N²) total.

~38K params with hidden_dim=64.
"""

import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass

from envs.gnm_env import NODE_FEAT_DIM, GRAPH_FEAT_DIM


@dataclass
class PolicyConfig:
    """Configuration for GNN edge selection policy."""
    hidden_dim: int = 64
    num_gnn_layers: int = 3


class GNNLayer(nn.Module):
    """D⁻¹A message passing with residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n, dim) node features
            adj_norm: (n, n) D⁻¹A normalized adjacency
        Returns:
            (n, dim) updated node features
        """
        msg = adj_norm @ x           # (n, dim) — O(N²d)
        return self.act(self.W(msg)) + x  # residual


class EdgeSelectionPolicy(nn.Module):
    """
    GNN policy with learned edge selection.

    The GNN encoder propagates structural information across the graph via
    message passing on the adjacency matrix. Edge scorers then use rich
    node embeddings (not just local degree stats) to decide which edges to swap.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # Input projection: node features → hidden dim
        self.input_proj = nn.Linear(NODE_FEAT_DIM, d)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(d) for _ in range(config.num_gnn_layers)
        ])

        # Edge scorers: concat(emb[i], emb[j], graph_feat) → score
        scorer_in = 2 * d + GRAPH_FEAT_DIM
        self.remove_scorer = nn.Sequential(
            nn.Linear(scorer_in, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.add_scorer = nn.Sequential(
            nn.Linear(scorer_in, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

        # Value head: mean_pool(emb) + graph_feat → value
        value_in = d + GRAPH_FEAT_DIM
        self.value_head = nn.Sequential(
            nn.Linear(value_in, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(
        self,
        node_feat: torch.Tensor,  # (n, NODE_FEAT_DIM)
        adj: torch.Tensor,        # (n, n)
    ) -> torch.Tensor:
        """Run GNN encoder. Returns (n, hidden_dim) node embeddings."""
        # D⁻¹A normalization
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg  # (n, n)

        x = self.input_proj(node_feat)  # (n, hidden_dim)
        for layer in self.gnn_layers:
            x = layer(x, adj_norm)
        return x

    def _score_edges(
        self,
        emb: torch.Tensor,         # (n, hidden_dim)
        indices: torch.Tensor,     # (P, 2) edge indices
        graph_feat: torch.Tensor,  # (GRAPH_FEAT_DIM,)
        scorer: nn.Module,
    ) -> torch.Tensor:
        """Score candidate edges using node embeddings. Returns (P,) logits."""
        P = indices.shape[0]
        ei = emb[indices[:, 0]]  # (P, hidden_dim)
        ej = emb[indices[:, 1]]  # (P, hidden_dim)
        gf_expanded = graph_feat.unsqueeze(0).expand(P, -1)  # (P, GRAPH_FEAT_DIM)
        x = torch.cat([ei, ej, gf_expanded], dim=1)  # (P, 2*hidden_dim + GRAPH_FEAT_DIM)
        return scorer(x).squeeze(-1)  # (P,)

    def act(
        self,
        node_feat: torch.Tensor,   # (n, NODE_FEAT_DIM)
        adj: torch.Tensor,         # (n, n)
        edges: torch.Tensor,       # (M, 2) edge indices
        non_edges: torch.Tensor,   # (NE, 2) non-edge indices
        graph_feat: torch.Tensor,  # (GRAPH_FEAT_DIM,)
        R: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action: which R edges to swap.

        Returns:
            remove_idx: (R,) indices into edges to remove
            add_idx: (R,) indices into non-edges to add
            log_prob: scalar total log probability
            value: scalar state value
        """
        emb = self._encode(node_feat, adj)  # (n, hidden_dim)

        # Value from mean-pooled embeddings
        graph_emb = emb.mean(dim=0)  # (hidden_dim,)
        value_input = torch.cat([graph_emb, graph_feat])  # (hidden_dim + GRAPH_FEAT_DIM,)
        value = self.value_head(value_input).squeeze(-1)

        M = edges.shape[0]
        NE = non_edges.shape[0]
        R = min(R, M, NE)

        if R == 0:
            dev = node_feat.device
            return (
                torch.zeros(0, dtype=torch.long, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev),
                torch.tensor(0.0, device=dev),
                value,
            )

        # Score edges for removal
        remove_logits = self._score_edges(emb, edges, graph_feat, self.remove_scorer)
        # Score non-edges for addition
        add_logits = self._score_edges(emb, non_edges, graph_feat, self.add_scorer)

        if deterministic:
            _, remove_idx = remove_logits.topk(R)
            _, add_idx = add_logits.topk(R)
        else:
            remove_probs = torch.softmax(remove_logits, dim=0)
            add_probs = torch.softmax(add_logits, dim=0)
            remove_idx = torch.multinomial(remove_probs, R, replacement=False)
            add_idx = torch.multinomial(add_probs, R, replacement=False)

        # Log probability
        log_remove = torch.log_softmax(remove_logits, dim=0)[remove_idx].sum()
        log_add = torch.log_softmax(add_logits, dim=0)[add_idx].sum()
        log_prob = log_remove + log_add

        return remove_idx, add_idx, log_prob, value

    def evaluate(
        self,
        node_feat: torch.Tensor,   # (n, NODE_FEAT_DIM)
        adj: torch.Tensor,         # (n, n)
        edges: torch.Tensor,       # (M, 2) edge indices
        non_edges: torch.Tensor,   # (NE, 2) non-edge indices
        graph_feat: torch.Tensor,  # (GRAPH_FEAT_DIM,)
        remove_idx: torch.Tensor,  # (R,) stored remove indices
        add_idx: torch.Tensor,     # (R,) stored add indices
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under current policy parameters (for PPO).

        Returns:
            log_prob: scalar total log probability
            entropy: scalar entropy
            value: scalar state value
        """
        emb = self._encode(node_feat, adj)

        # Value
        graph_emb = emb.mean(dim=0)
        value_input = torch.cat([graph_emb, graph_feat])
        value = self.value_head(value_input).squeeze(-1)

        if len(remove_idx) == 0:
            dev = node_feat.device
            return (
                torch.tensor(0.0, device=dev),
                torch.tensor(0.0, device=dev),
                value,
            )

        # Re-score edges under current parameters
        remove_logits = self._score_edges(emb, edges, graph_feat, self.remove_scorer)
        add_logits = self._score_edges(emb, non_edges, graph_feat, self.add_scorer)

        log_remove = torch.log_softmax(remove_logits, dim=0)[remove_idx].sum()
        log_add = torch.log_softmax(add_logits, dim=0)[add_idx].sum()
        log_prob = log_remove + log_add

        # Entropy
        remove_lsm = torch.log_softmax(remove_logits, dim=0)
        add_lsm = torch.log_softmax(add_logits, dim=0)
        entropy_remove = -(torch.softmax(remove_logits, dim=0) * remove_lsm).sum()
        entropy_add = -(torch.softmax(add_logits, dim=0) * add_lsm).sum()
        entropy = entropy_remove + entropy_add

        return log_prob, entropy, value
