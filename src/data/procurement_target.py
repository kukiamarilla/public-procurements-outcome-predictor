"""Convención de etiqueta binaria a partir del dataset de licitaciones (OCDS / status)."""

from __future__ import annotations


def training_y_from_procurement_row(row: dict) -> float | None:
    """
    Target desde `status` (o `tenderStatus`):

    - 1.0 — exitoso: ``complete``
    - 0.0 — no exitoso: ``unsuccessful``, ``cancelled`` o ``canceled``
    - None — cualquier otro valor (p. ej. ``planning``, ``active``): sin etiqueta para entrenamiento
    """
    st = str(row.get("status") or row.get("tenderStatus") or "").strip().lower()
    if st == "complete":
        return 1.0
    if st in ("unsuccessful", "cancelled", "canceled"):
        return 0.0
    return None
