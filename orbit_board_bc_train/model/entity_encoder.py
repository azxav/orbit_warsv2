from __future__ import annotations

import torch
from torch import nn


class EntityEncoder(nn.Module):
    def __init__(self, planet_dim: int, fleet_dim: int, global_dim: int, hidden_dim: int):
        super().__init__()
        self.planet = nn.Sequential(nn.Linear(planet_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.fleet = nn.Sequential(nn.Linear(fleet_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.global_token = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, planet_tokens: torch.Tensor, fleet_tokens: torch.Tensor, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.planet(planet_tokens), self.fleet(fleet_tokens), self.global_token(global_features).unsqueeze(1)

