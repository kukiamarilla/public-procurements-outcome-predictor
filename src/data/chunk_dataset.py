from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from data.procurement_target import training_y_from_procurement_row


def list_labeled_embedding_paths_from_dataset_json(
    cache_dir: str,
    dataset_json_path: str,
) -> Tuple[list[str], list[float]]:
    """
    Empareja cada `.pt` en `cache_dir` con una fila del JSON por `tender_id` en el checkpoint
    y toma `y` desde `status` (vía `training_y_from_procurement_row`). Ignora licitaciones
    cuyo status no sea complete / unsuccessful / cancelled.
    """
    raw = Path(dataset_json_path).expanduser().resolve().read_text(encoding="utf-8")
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("El dataset JSON debe ser un array de objetos")

    tid_to_y: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("tenderId", "")).strip()
        if not tid:
            continue
        y = training_y_from_procurement_row(row)
        if y is not None:
            tid_to_y[tid] = y

    root = os.path.abspath(cache_dir)
    paths: list[str] = []
    labels: list[float] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(root, name)
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        tid = str(d.get("tender_id") or "").strip()
        if not tid or tid not in tid_to_y:
            continue
        paths.append(path)
        labels.append(tid_to_y[tid])
    return paths, labels


def list_labeled_embedding_paths(cache_dir: str) -> Tuple[list[str], list[float]]:
    """
    Recorre `cache_dir`, devuelve rutas absolutas de .pt que incluyen la clave `y` (etiqueta de entrenamiento).
    """
    root = os.path.abspath(cache_dir)
    paths: list[str] = []
    labels: list[float] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(root, name)
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        if d.get("y") is None:
            continue
        paths.append(path)
        labels.append(float(d["y"]))
    return paths, labels


class CachedChunkEmbDataset(Dataset):
    """
    Un .pt por muestra, generado aguas arriba (PDF → texto → chunks → embedder):

    {
      "embs": Tensor [N, d_in],
      "y": float en [0,1] opcional; si falta, el Dataset devuelve y = -1 (filtrar en entrenamiento)
    }

    Si pasás ``y_list`` (misma longitud que ``files``), se usa esa etiqueta en lugar de la del .pt
    (p. ej. alineada con ``procurements_dataset.json``).
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        files: list[str] | None = None,
        y_list: list[float] | None = None,
    ):
        if files is not None:
            self.files = list(files)
        elif cache_dir is not None:
            self.files = sorted(
                os.path.join(cache_dir, f)
                for f in os.listdir(cache_dir)
                if f.endswith(".pt")
            )
        else:
            raise ValueError("CachedChunkEmbDataset requiere cache_dir o files.")
        self._y_list = list(y_list) if y_list is not None else None
        if self._y_list is not None and len(self._y_list) != len(self.files):
            raise ValueError("y_list debe tener la misma longitud que files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(path, map_location="cpu")
        embs = d["embs"].float()
        if self._y_list is not None:
            y = torch.tensor(float(self._y_list[idx]), dtype=torch.float32)
        else:
            raw_y = d.get("y")
            y = torch.tensor(float(raw_y), dtype=torch.float32) if raw_y is not None else torch.tensor(
                -1.0,
                dtype=torch.float32,
            )
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
