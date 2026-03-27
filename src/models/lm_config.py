"""Configuración del LM causal (embedder) + hiperparámetros del predictor acoplados."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    model_id: str = "openai/gpt-oss-20b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Chunking
    max_len: int = 4096
    stride: int = 2048

    # Cross-chunk encoder (TenderSuccessPredictor)
    d_model: int = 512
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    num_layers: int = 1
