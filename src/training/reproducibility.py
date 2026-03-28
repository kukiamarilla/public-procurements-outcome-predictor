"""Semillas y opciones deterministas para repetir entrenamientos con los mismos hiperparámetros."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def _set_cuda_deterministic_env() -> None:
    # Mejora determinismo en funciones BLAS en GPU (documentación PyTorch / cuBLAS).
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def configure_reproducibility(seed: int) -> None:
    """
    Fija semillas de random / NumPy / PyTorch y cuDNN en modo determinista.

    Llamá esto al inicio del proceso y **otra vez al comenzar cada fold** con una
    semilla distinta (p. ej. ``base + fold * K``) para que el init del modelo y el
    orden de batches no dependan del estado del RNG al terminar el fold anterior.
    """
    _set_cuda_deterministic_env()
    s = int(seed)
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloader_worker_init_fn(base_seed: int):
    """``worker_init_fn`` para ``DataLoader`` con ``num_workers > 0``."""

    def _fn(worker_id: int) -> None:
        w = int(base_seed) + int(worker_id)
        random.seed(w)
        np.random.seed(w % (2**32 - 1))
        torch.manual_seed(w % (2**32 - 1))

    return _fn
