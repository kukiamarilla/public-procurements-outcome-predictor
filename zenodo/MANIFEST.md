# Zenodo Package Manifest

## Included Directories

- `configs/`: non-secret configuration conventions for local and remote paths.
- `paper/`: LaTeX paper source, PDF, figures, paper scripts, revision checklist, camera-ready notes, and funnel documentation.
- `scripts/`: ETL, synchronization, baseline, and training entry points.
- `src/`: reusable Python modules for configuration, data loading, PDF extraction, models, and training utilities.
- `zenodo/`: this upload documentation.

## Included Root Files

- `.env.example`
- `.gitignore`
- `CITATION.cff`
- `README.md`
- `pyproject.toml`
- `uv.lock`

## Excluded Local Data and Generated Files

The package excludes local data directories because the project treats DigitalOcean Spaces as the authoritative storage layer:

- `data/raw/**`
- `data/processed/**`
- `data/chunk_embeddings/**`
- `data/mlflow.db`
- `outputs/**`
- `experiments/**`
- `mlruns/**`
- `mlartifacts/**`

It also excludes local secrets and transient files:

- `.env`
- `.venv/**`
- Python cache directories
- LaTeX auxiliary files such as `.aux`, `.log`, `.out`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`, `.synctex.gz`
- OS/editor metadata

## Local Data Role Confirmed From Code

- `scripts/etl/*` read and write the canonical dataset artifacts in Spaces, then optionally write local JSON checkpoints.
- `scripts/sync_embeddings_for_training.py` downloads `.pt` embeddings from Spaces into `data/chunk_embeddings` only when local training is needed.
- `scripts/train_cv_mlflow.py` trains from local cached embeddings because PyTorch consumes local `.pt` files, while MLflow artifacts are configured to use Spaces.
- TF-IDF and paper scripts cache PBC text under `data/raw/pbc_txt_cache` only to avoid repeatedly downloading text objects from Spaces.

Therefore, local `data/` contents are reproducible caches or snapshots and should not be archived as primary data in Zenodo.
