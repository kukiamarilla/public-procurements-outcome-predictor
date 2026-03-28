"""Hiperparámetros y rutas de entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformer sobre chunks
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.3
    num_layers: int = 1

    # Entrenamiento
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 50
    weight_decay: float = 0.01

    # Caché de .pt por licitación (embs + y); ver data.chunk_dataset.CachedChunkEmbDataset
    cache_dir: str = "data/chunk_embeddings"


def default_cfg() -> CFG:
    """Instancia fresca de configuración (device recalculado al construir)."""
    return CFG()
