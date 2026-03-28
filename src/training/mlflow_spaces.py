"""
MLflow + DigitalOcean Spaces (S3-compatible): artefactos bajo …/outcome-predictor/mlflow.

Las métricas y parámetros siguen el tracking store configurado en `MLFLOW_TRACKING_URI`
(por defecto el script usa SQLite local en `data/mlflow.db`). Solo los artefactos pesados
(modelos, ficheros) se suben al bucket al crear el experimento con `artifact_location` S3.
"""

from __future__ import annotations

import os


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def spaces_bucket_name() -> str | None:
    b = _env("DO_SPACES_BUCKET", "SPACES_BUCKET")
    return b if b else None


def _dataset_prefix_key() -> str:
    """Igual que `spaces_io.dataset_prefix()` (sin importar scripts ETL)."""
    explicit = _env("DO_SPACES_DATASET_PREFIX", "DO_SPACES_PREFIX_INPUT")
    if explicit:
        return explicit.strip().strip("/")
    global_prefix = os.environ.get("SPACES_PREFIX")
    p = (global_prefix or "").strip().strip("/")
    r = "outcome-predictor/procurements"
    return f"{p}/{r}" if p else r


def spaces_mlflow_s3_prefix_key() -> str:
    """
    Prefijo objeto bajo el bucket: hermano de `…/procurements` → `…/outcome-predictor/mlflow`.

    Override: variable `SPACES_MLFLOW_PREFIX` (ruta completa bajo bucket, sin s3://).
    """
    override = _env("SPACES_MLFLOW_PREFIX")
    if override:
        return override.strip().strip("/")
    dp = _dataset_prefix_key().rstrip("/")
    suf = "/procurements"
    if dp.endswith(suf):
        return f"{dp[: -len(suf)]}/mlflow"
    return "outcome-predictor/mlflow"


def spaces_mlflow_artifact_root() -> str | None:
    """URI `s3://bucket/.../outcome-predictor/mlflow` o None si falta bucket."""
    bucket = spaces_bucket_name()
    if not bucket:
        return None
    key = spaces_mlflow_s3_prefix_key()
    return f"s3://{bucket}/{key}"


def configure_mlflow_s3_env_from_spaces() -> None:
    """
    Propaga credenciales Spaces a las variables que consumen MLflow y botocore al escribir en S3.

    No pisa valores ya definidos en el entorno (podés forzar AWS_* / MLFLOW_S3_* a mano).
    """
    endpoint = _env("DO_SPACES_ENDPOINT", "SPACES_ENDPOINT")
    access = _env("DO_SPACES_ACCESS_KEY", "SPACES_ACCESS_KEY")
    secret = _env("DO_SPACES_SECRET_KEY", "SPACES_SECRET_KEY")
    region = _env("DO_SPACES_REGION", "SPACES_REGION")

    if endpoint and not os.environ.get("MLFLOW_S3_ENDPOINT_URL"):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint.rstrip("/")
    if endpoint and not os.environ.get("AWS_ENDPOINT_URL"):
        os.environ["AWS_ENDPOINT_URL"] = endpoint.rstrip("/")
    if access and not os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = access
    if secret and not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret
    if region and not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = region


def ensure_mlflow_experiment(
    mlflow: object,
    *,
    name: str,
    artifact_root: str | None,
) -> str:
    """
    Devuelve `experiment_id`. Crea el experimento con `artifact_root` S3 si aplica y aún no existe.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is None:
        if artifact_root:
            eid = client.create_experiment(name, artifact_location=artifact_root)
        else:
            eid = client.create_experiment(name)
        mlflow.set_experiment(experiment_id=eid)
        return str(eid)

    eid = existing.experiment_id
    loc = (existing.artifact_location or "").strip()
    want_s3 = bool(artifact_root and str(artifact_root).startswith("s3:"))
    if want_s3 and loc and not loc.startswith("s3:"):
        print(
            f"Aviso: el experimento «{name}» ya existe con artifact_location local «{loc}». "
            "Los artefactos no irán a Spaces hasta que borres el experimento en MLflow o uses "
            "otro nombre (--experiment).",
        )
    mlflow.set_experiment(experiment_id=eid)
    return str(eid)
