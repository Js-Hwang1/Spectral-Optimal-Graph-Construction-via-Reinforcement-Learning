"""
REFINE Policy Network — Edge-level MLP scorer.

Architecture:
  - ADD MLP:  (N,N,4) edge features -> (N,N) logits for adding non-edges
  - REM MLP:  (N,N,4) edge features -> (N,N) logits for removing edges
  - Value MLP: (3,) graph features -> scalar V(s)

Edge features (4-dim per pair):
  0: n * |v₂ᵢ - v₂ⱼ|²  — Fiedler gap (r=0.90 with Δλ₂)
  1: degᵢ / (n-1)       — source degree normalized
  2: degⱼ / (n-1)       — target degree normalized
  3: step / K           — budget awareness

Graph features (3-dim):
  0: mean_degree / (n-1)
  1: λ₂ / n
  2: step / K

Ablation PoC showed v3/v4/v5 gaps (r<0.15) and spectral gap scalars
add noise that hurts MLP edge selection, especially at mid-density.

O(N²) inference: MLP applied to N² pairs with fixed hidden dim.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


EDGE_FEAT_DIM = 4
GRAPH_FEAT_DIM = 3


@dataclass
class RefineConfig:
    """Configuration for REFINE policy network."""
    hidden_dim: int = 64
    edge_feat_dim: int = EDGE_FEAT_DIM
    graph_feat_dim: int = GRAPH_FEAT_DIM
    delta: int = 2
    temperature: float = 1.0


class RefinePolicy(nn.Module):
    """
    Edge-level MLP policy for REFINE.

    Scores each edge/non-edge independently using spectral features.
    No message passing — the Fiedler gap already encodes the key structural signal.
    """

    def __init__(self, config: RefineConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        ef = config.edge_feat_dim
        gf = config.graph_feat_dim

        self.add_mlp = nn.Sequential(
            nn.Linear(ef, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

        self.rem_mlp = nn.Sequential(
            nn.Linear(ef, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(gf, d),
            nn.SiLU(),
            nn.Linear(d, d),
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

    def score_add(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Score pairs for ADD phase.

        Args:
            edge_features: (..., F) edge features

        Returns:
            logits: (...) scores with last dim squeezed
        """
        shape = edge_features.shape[:-1]
        flat = edge_features.reshape(-1, self.config.edge_feat_dim)
        scores = self.add_mlp(flat).squeeze(-1)
        return scores.reshape(shape) / self.config.temperature

    def score_remove(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Score pairs for REMOVE phase.

        Args:
            edge_features: (..., F) edge features

        Returns:
            logits: (...) scores with last dim squeezed
        """
        shape = edge_features.shape[:-1]
        flat = edge_features.reshape(-1, self.config.edge_feat_dim)
        scores = self.rem_mlp(flat).squeeze(-1)
        return scores.reshape(shape) / self.config.temperature

    def compute_value(self, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value from graph-level features.

        Args:
            graph_features: (G,) graph features

        Returns:
            value: scalar
        """
        return self.value_mlp(graph_features).squeeze(-1)
