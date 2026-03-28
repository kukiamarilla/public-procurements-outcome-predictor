"""Modelo end-to-end: texto → ChunkEmbedder → TenderSuccessPredictor."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .embedder import (
    ChunkEmbedder,
    build_chunk_embedder,
    forward_text_resolving_cuda_oom,
    infer_input_dim,
)
from .lm_config import ModelConfig
from .predictor import TenderSuccessPredictor


class TenderSuccessModel(nn.Module):
    """
    1) Entrenamiento con embeddings cacheados:
           logits = model(chunk_embs=embs, valid_mask=mask)
    2) Inferencia desde texto:
           prob = model.predict_from_text(text)
    """

    def __init__(self, embedder: ChunkEmbedder, predictor: TenderSuccessPredictor):
        super().__init__()
        self.embedder = embedder
        self.predictor = predictor

    def forward(
        self,
        chunk_embs: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predictor(chunk_embs, valid_mask)

    @torch.no_grad()
    def predict_from_text(self, text: str) -> torch.Tensor:
        if str(self.embedder.device).startswith("cuda"):
            embs = forward_text_resolving_cuda_oom(self.embedder, text)
        else:
            embs = self.embedder(text)
        chunk_vecs = embs.float().unsqueeze(0)
        valid_mask = torch.ones(
            (1, chunk_vecs.size(1)),
            device=chunk_vecs.device,
            dtype=torch.bool,
        )
        return self.predictor.predict_proba(chunk_vecs, valid_mask).squeeze(0)


def build_model(config: ModelConfig) -> TenderSuccessModel:
    embedder = build_chunk_embedder(config)
    d_in = infer_input_dim(embedder)

    predictor = TenderSuccessPredictor(
        d_in=d_in,
        d_model=config.d_model,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        num_layers=config.num_layers,
    ).to(config.device)

    return TenderSuccessModel(embedder=embedder, predictor=predictor)
