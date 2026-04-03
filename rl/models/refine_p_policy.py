"""
REFINE+ Policy Network — 10-dim rich spectral features.

Subsumes both FV (Fiedler gap) and ER (effective resistance) information,
plus structural features (common neighbors) and spectral stability signals
(λ₃-λ₂ gap, λ₄-λ₂ width) that neither baseline uses.

Edge features (10-dim per pair):
  0: n * |v₂ᵢ - v₂ⱼ|²       — Fiedler gap (what FV uses)
  1: n * |v₃ᵢ - v₃ⱼ|²       — secondary bottleneck gap
  2: R(i,j) / R_mean         — effective resistance (what ER uses)
  3: cn(i,j) / (n-2)         — common neighbors fraction
  4: degᵢ / (n-1)            — source degree
  5: degⱼ / (n-1)            — target degree
  6: (λ₃ - λ₂) / n           — spectral gap stability
  7: (λ₄ - λ₂) / n           — spectral width
  8: step / K                 — budget awareness
  9: s / num_swaps            — swap progress within step

Graph features (7-dim):
  0: mean_degree / (n-1)
  1: λ₂ / n
  2: λ₃ / n
  3: λ₄ / n
  4: (λ₃ - λ₂) / n
  5: R_mean * n
  6: step / K

~20K params with hidden_dim=64.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


EDGE_FEAT_DIM = 10
GRAPH_FEAT_DIM = 7


@dataclass
class RefinePConfig:
    """Configuration for REFINE+ policy network."""
    hidden_dim: int = 64
    edge_feat_dim: int = EDGE_FEAT_DIM
    graph_feat_dim: int = GRAPH_FEAT_DIM
    temperature: float = 1.0


class RefinePPolicy(nn.Module):
    """
    REFINE+ edge-level MLP policy with rich spectral features.

    Same architecture as REFINE v7 but with 10-dim input features that
    subsume FV (Fiedler gap) and ER (effective resistance) information.
    """

    def __init__(self, config: RefinePConfig):
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
        shape = edge_features.shape[:-1]
        flat = edge_features.reshape(-1, self.config.edge_feat_dim)
        scores = self.add_mlp(flat).squeeze(-1)
        return scores.reshape(shape) / self.config.temperature

    def score_remove(self, edge_features: torch.Tensor) -> torch.Tensor:
        shape = edge_features.shape[:-1]
        flat = edge_features.reshape(-1, self.config.edge_feat_dim)
        scores = self.rem_mlp(flat).squeeze(-1)
        return scores.reshape(shape) / self.config.temperature

    def compute_value(self, graph_features: torch.Tensor) -> torch.Tensor:
        return self.value_mlp(graph_features).squeeze(-1)
