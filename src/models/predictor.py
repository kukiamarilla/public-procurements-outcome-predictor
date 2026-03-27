from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import CFG


class TenderSuccessPredictor(nn.Module):
    """
    Probabilidad de éxito / outcome a partir de embeddings por chunk.

    Entrada (post pipeline PDF → texto → chunking → embedder, guardado en .pt):
      - chunk_embs: [B, N, d_in]
      - valid_mask: [B, N] True en chunks válidos

    Salida:
      - logits: [B, 1]  → prob = sigmoid(logits)
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()

        self.proj = nn.Linear(d_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        chunk_embs: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, _ = chunk_embs.shape

        x = self.proj(chunk_embs)

        cls = self.cls.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)

        if valid_mask is not None:
            cls_valid = torch.ones((batch_size, 1), device=x.device, dtype=torch.bool)
            valid = torch.cat([cls_valid, valid_mask], dim=1)
            pad_mask = ~valid
        else:
            pad_mask = None

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = x[:, 0]

        return self.head(pooled)

    @torch.no_grad()
    def predict_proba(
        self,
        chunk_embs: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        logits = self.forward(chunk_embs, valid_mask)
        return torch.sigmoid(logits)


def build_model_from_sample_batch(sample_embs: torch.Tensor, cfg: CFG) -> TenderSuccessPredictor:
    """Crea el predictor con d_in inferido de tensores [B,N,d] o [N,d]."""
    if sample_embs.dim() == 3:
        d_in = sample_embs.shape[-1]
    elif sample_embs.dim() == 2:
        d_in = sample_embs.shape[-1]
    else:
        raise ValueError("sample_embs debe ser [B,N,d] o [N,d]")

    return TenderSuccessPredictor(
        d_in=d_in,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        num_layers=cfg.num_layers,
    ).to(cfg.device)
