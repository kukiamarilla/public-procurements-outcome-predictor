# Zenodo Upload Notes

This archive is intended to preserve the code, paper artifacts, and reproducibility documentation for the thesis project "Public Procurements Outcome Predictor".

## What Is Included

- Source code under `src/`.
- Experiment and ETL scripts under `scripts/`.
- Configuration templates under `configs/` and `.env.example`.
- Paper source, paper figures, paper scripts, and funnel documentation under `paper/`.
- Project dependency files: `pyproject.toml` and `uv.lock`.
- Citation metadata in `CITATION.cff`.

## What Is Not Included

The archive intentionally excludes local data and local experiment outputs:

- `data/raw/`
- `data/processed/`
- `data/chunk_embeddings/`
- `data/mlflow.db`
- `outputs/`
- `experiments/`
- `mlruns/`
- `.env`

In this project, these local paths are caches, snapshots, or generated outputs. The data source of truth is DigitalOcean Spaces, configured through the `SPACES_*` variables documented in `.env.example`.

## Remote Data Layout

The code expects the remote dataset and generated artifacts under the Spaces prefix configured by `SPACES_PREFIX`, with project paths documented in `configs/data.yaml`.

Main remote objects and prefixes:

- `outcome-predictor/procurements/procurements.json`
- `outcome-predictor/procurements/ids_unsuccessful.json`
- `outcome-predictor/procurements/ids_cancelled.json`
- `outcome-predictor/procurements/procurements_dataset.json`
- `outcome-predictor/pbcs/pdf/`
- `outcome-predictor/pbcs/txt/`
- `outcome-predictor/pbcs/embeddings/`
- `outcome-predictor/mlflow/`

The processing funnel and failure modes are documented in `paper/funnel.txt`.

## Reproducibility Outline

1. Create a local environment:

   ```bash
   uv sync --extra train --extra pdf
   ```

2. Copy `.env.example` to `.env` and set the `SPACES_*` variables.

3. Sync embeddings from Spaces when local training is needed:

   ```bash
   uv run python scripts/sync_embeddings_for_training.py --out-dir data/chunk_embeddings
   ```

4. Run cross-validation training:

   ```bash
   uv run python scripts/train_cv_mlflow.py \
     --cache-dir data/chunk_embeddings \
     --dataset-json data/processed/procurements_dataset.json \
     --folds 5
   ```

5. For the paper-specific analyses, see the scripts under `paper/scripts/`.

## Suggested Zenodo Metadata

- Upload type: Software
- Title: Public Procurements Outcome Predictor
- Creator: Kuki Amarilla
- Related identifier: `https://github.com/kukiamarilla/public-procurements-outcome-predictor`
- Keywords: public procurement, eGovernment, document classification, LLM embeddings, reproducible research
- Description: Code, paper artifacts, and reproducibility documentation for predicting public procurement outcomes from bidding documents in Paraguay.

Before publishing, choose the license explicitly in Zenodo. This repository currently does not include a license file.
