from __future__ import annotations

import torch
from torch import nn


class SetTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=padding_mask)

