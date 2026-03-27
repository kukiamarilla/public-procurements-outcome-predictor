from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class CachedChunkEmbDataset(Dataset):
    """
    Un .pt por muestra, generado aguas arriba (PDF → texto → chunks → embedder):

    {
      "embs": Tensor [N, d_in],
      "y": float en [0,1]   # etiqueta dura o soft
    }
    """

    def __init__(self, cache_dir: str):
        self.files = sorted(
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        embs = d["embs"].float()
        y = torch.tensor(d["y"], dtype=torch.float32)
        return embs, y


def collate_pad_chunks(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lista (embs [N_i,d], y escalar) → (embs [B,Nmax,d], valid [B,Nmax], y [B,1])."""
    embs_list, y_list = zip(*batch)

    B = len(embs_list)
    d = embs_list[0].shape[1]
    n_max = max(e.shape[0] for e in embs_list)

    embs = torch.zeros((B, n_max, d), dtype=torch.float32)
    valid = torch.zeros((B, n_max), dtype=torch.bool)

    for i, e in enumerate(embs_list):
        n = e.shape[0]
        embs[i, :n] = e
        valid[i, :n] = True

    y = torch.stack(y_list).view(B, 1).float()
    return embs, valid, y
