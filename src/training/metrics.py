from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        roc_auc_score,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Instalá scikit-learn: uv sync --extra train",
    ) from e


def binary_classification_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Métricas para etiquetas binarias (0/1). No incluye accuracy como métrica principal.

    - ROC AUC, PR AUC (= average precision), Brier: sobre probabilidades.
    - F1 y balanced accuracy: predicción dura con `threshold` sobre la prob. de clase positiva.

    **PR AUC y prevalencia:** en clasificación binaria desbalanceada, el PR AUC de un clasificador
    sin habilidad con orden aleatorio de scores suele situarse en la línea horizontal de precisión
    igual a la **fracción de positivos** en el conjunto. Por eso reportamos
    ``positive_prevalence`` (π) y ``pr_auc_excess_over_prevalence`` = PR_AUC − π como lectura
    rápida de cuánto aporta el modelo respecto a ese referente frecuente en la literatura.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    p_clip = np.clip(p, 1e-7, 1.0 - 1e-7)

    out: dict[str, Any] = {}
    pi = float(np.mean(y_true))
    out["positive_prevalence"] = pi
    out["pr_auc_prevalence_baseline"] = pi

    classes = np.unique(y_true)
    if len(classes) >= 2:
        pr = float(average_precision_score(y_true, p))
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = pr
        out["pr_auc_excess_over_prevalence"] = float(pr - pi)
        y_hat = (p >= float(threshold)).astype(np.int64)
        y_bin = y_true.astype(np.int64)
        out["f1"] = float(f1_score(y_bin, y_hat, zero_division=0))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_bin, y_hat))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["pr_auc_excess_over_prevalence"] = float("nan")
        out["f1"] = float("nan")
        out["balanced_accuracy"] = float("nan")

    out["log_loss"] = float(log_loss(y_true, p_clip))
    out["brier_score"] = float(brier_score_loss(y_true, p))
    return out
