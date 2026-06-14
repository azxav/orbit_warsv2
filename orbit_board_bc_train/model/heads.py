from __future__ import annotations

import torch
from torch import nn


class PointerHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.source_query = nn.Linear(hidden_dim, hidden_dim)
        self.target_query = nn.Linear(hidden_dim, hidden_dim)
        self.stop_key = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.angle_offset = nn.Linear(hidden_dim, 1)
        self.ship_fraction = nn.Linear(hidden_dim, 1)

    def forward(self, decoded: torch.Tensor, planet_memory: torch.Tensor) -> dict[str, torch.Tensor]:
        source_q = self.source_query(decoded)
        target_q = self.target_query(decoded)
        planet_keys = planet_memory.transpose(1, 2)
        source_planet_logits = torch.matmul(source_q, planet_keys)
        stop_logits = torch.einsum("bah,h->ba", source_q, self.stop_key).unsqueeze(-1)
        source_logits = torch.cat([source_planet_logits, stop_logits], dim=-1)
        target_logits = torch.matmul(target_q, planet_keys)
        return {
            "source_logits": source_logits,
            "target_logits": target_logits,
            "angle_offset": self.angle_offset(decoded).squeeze(-1),
            "ship_fraction": torch.sigmoid(self.ship_fraction(decoded).squeeze(-1)),
        }

