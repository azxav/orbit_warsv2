from __future__ import annotations

import torch
from torch import nn

from .action_decoder import ActionDecoder
from .entity_encoder import EntityEncoder
from .graph_transformer import SetTransformerEncoder
from .heads import PointerHeads


class BoardBCModel(nn.Module):
    def __init__(
        self,
        planet_dim: int,
        fleet_dim: int,
        global_dim: int,
        hidden_dim: int = 192,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        heads: int = 6,
        dropout: float = 0.05,
        max_actions: int = 32,
        max_planets: int = 64,
    ):
        super().__init__()
        self.max_planets = max_planets
        self.max_actions = max_actions
        self.entity_encoder = EntityEncoder(planet_dim, fleet_dim, global_dim, hidden_dim)
        self.board_encoder = SetTransformerEncoder(hidden_dim, encoder_layers, heads, dropout)
        self.decoder = ActionDecoder(hidden_dim, decoder_layers, heads, max_actions, dropout)
        self.heads = PointerHeads(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        planet_mem, fleet_mem, global_mem = self.entity_encoder(
            batch["planet_tokens"], batch["fleet_tokens"], batch["global_features"]
        )
        memory = torch.cat([global_mem, planet_mem, fleet_mem], dim=1)
        global_mask = torch.zeros((memory.shape[0], 1), dtype=torch.bool, device=memory.device)
        memory_padding_mask = torch.cat([global_mask, ~batch["planet_masks"], ~batch["fleet_masks"]], dim=1)
        encoded = self.board_encoder(memory, memory_padding_mask)
        planet_encoded = encoded[:, 1 : 1 + self.max_planets]
        decoded = self.decoder(encoded, memory_padding_mask)
        outputs = self.heads(decoded, planet_encoded)
        if "source_candidate_mask" in batch:
            outputs["source_logits"] = outputs["source_logits"].masked_fill(~batch["source_candidate_mask"], -1e9)
        if "target_candidate_mask" in batch:
            target_mask = batch["target_candidate_mask"]
            outputs["target_logits"] = outputs["target_logits"].masked_fill(~target_mask, -1e9)
        return outputs

