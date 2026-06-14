from __future__ import annotations

import torch
from torch import nn


class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, max_actions: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(max_actions, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.max_actions = max_actions

    def forward(self, memory: torch.Tensor, memory_padding_mask: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        tgt = self.query.unsqueeze(0).expand(batch, -1, -1)
        causal_mask = torch.triu(torch.ones(self.max_actions, self.max_actions, device=memory.device, dtype=torch.bool), diagonal=1)
        return self.decoder(tgt, memory, tgt_mask=causal_mask, memory_key_padding_mask=memory_padding_mask)

