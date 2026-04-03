"""
REFINE Policy Network — Edge-level MLP scorer.

Copy of rl/models/refine_policy.py for CRL standalone usage.
Checkpoint-compatible with existing eval_refine.py.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


EDGE_FEAT_DIM = 4
GRAPH_FEAT_DIM = 3


@dataclass
class RefineConfig:
    hidden_dim: int = 64
    edge_feat_dim: int = EDGE_FEAT_DIM
    graph_feat_dim: int = GRAPH_FEAT_DIM
    delta: int = 2
    temperature: float = 1.0


class RefinePolicy(nn.Module):
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
