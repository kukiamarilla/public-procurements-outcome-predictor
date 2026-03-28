"""
Compara runs de un experimento MLflow con gráficos (métricas `cv_mean_*` y barras de error `cv_std_*`).

  uv sync --extra train
  MLFLOW_TRACKING_URI=sqlite:///$(pwd)/data/mlflow.db \\
    uv run python scripts/plot_mlflow_metrics.py --experiment procurements_predictor

  uv run python scripts/plot_mlflow_metrics.py \\
    --experiment procurements_predictor --out figures/metrics_runs.png --show

Solo se incluyen runs **padre** (sin `mlflow.parentRunId`), típicos del CV en train_cv_mlflow.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gráficos de métricas MLflow entre runs.")
    p.add_argument(
        "--experiment",
        type=str,
        default="procurements_predictor",
        help="Nombre del experimento en MLflow.",
    )
    p.add_argument(
        "--tracking-uri",
        type=str,
        default="",
        help="Override de MLFLOW_TRACKING_URI (por defecto: env o sqlite en data/mlflow.db).",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="roc_auc,pr_auc,f1,balanced_accuracy,brier_score,log_loss",
        help="Claves cortas; se leen cv_mean_<clave> y cv_std_<clave>.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "figures" / "mlflow_cv_metrics.png",
        help="Ruta del PNG (crea directorios si hace falta).",
    )
    p.add_argument("--show", action="store_true", help="Abrir ventana interactiva además de guardar.")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--max-runs", type=int, default=50, help="Máximo de runs padre a traer (orden cronológico).")
    return p.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
        from mlflow.tracking import MlflowClient
    except ImportError as e:
        raise SystemExit("Instalá mlflow y matplotlib: uv sync --extra train") from e

    from dotenv import load_dotenv

    load_dotenv()
    args = _parse_args()

    uri = (args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if not uri:
        db = (REPO_ROOT / "data" / "mlflow.db").resolve()
        if not db.is_file():
            raise SystemExit(
                f"No hay MLFLOW_TRACKING_URI y no existe {db}. "
                "Definí MLFLOW_TRACKING_URI o pasá --tracking-uri.",
            )
        uri = f"sqlite:///{db.as_posix()}"

    metric_suffixes = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_suffixes:
        raise SystemExit("Lista --metrics vacía.")

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"No existe el experimento «{args.experiment}» en {uri}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=max(500, args.max_runs * 10),
        order_by=["start_time ASC"],
    )

    parents: list = []
    for r in runs:
        if r.data.tags.get("mlflow.parentRunId"):
            continue
        parents.append(r)
    parents = parents[-args.max_runs :]

    if not parents:
        raise SystemExit("No hay runs padre en ese experimento (o todos son nested).")

    labels: list[str] = []
    for r in parents:
        name = r.data.tags.get("mlflow.runName") or r.info.run_id[:8]
        labels.append(str(name))

    n_met = len(metric_suffixes)
    ncols = min(3, n_met)
    nrows = (n_met + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)

    x = range(len(labels))
    for ax_idx, suffix in enumerate(metric_suffixes):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        mean_key = f"cv_mean_{suffix}"
        std_key = f"cv_std_{suffix}"
        means: list[float] = []
        stds: list[float] = []
        for r in parents:
            m = r.data.metrics.get(mean_key)
            s = r.data.metrics.get(std_key)
            means.append(float(m) if m is not None else float("nan"))
            stds.append(float(s) if s is not None else 0.0)

        colors = plt.cm.tab10(range(len(means)))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(mean_key.replace("cv_mean_", ""))
        ax.set_ylabel("valor")
        ax.grid(axis="y", alpha=0.3)

    for ax_idx in range(len(metric_suffixes), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Experiment «{args.experiment}» · {len(parents)} runs padre · {uri[:48]}…", fontsize=10)
    fig.tight_layout()

    args.out = args.out.expanduser().resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Guardado: {args.out}", file=sys.stderr)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
