from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(1, layers - 1)):
            modules.extend([nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            current = hidden_dim
        modules.append(nn.Linear(current, output_dim, bias=False))
        self.network = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class PredictorMLP(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MobileEmbeddingHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(value), dim=-1)
