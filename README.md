# Public procurements outcome predictor

## Entorno

```bash
cp .env.example .env
# Editar .env con tu region, endpoint, keys y bucket de DigitalOcean Spaces

uv sync
```

Los datos viven en **Spaces**; `data/raw` y `data/processed` son caché o artefactos locales opcionales (ver `.gitignore`).

### Código del modelo (bajo `src/`)

El paquete instalable expone módulos de primer nivel: `config`, `models`, `data`, `training` (sin carpeta extra con el nombre del proyecto: todo vive directamente en `src/`).

Código PyTorch del predictor sobre **embeddings por chunk** (PDF → texto → chunks → embedder → `.pt` con `embs` y `y`, ver `data.CachedChunkEmbDataset`).

```bash
uv sync
uv run python -c "from config import CFG; from models import TenderSuccessPredictor; print(TenderSuccessPredictor)"
```

Imports típicos: `config.CFG`, `models.TenderSuccessPredictor`, `data.collate_pad_chunks`, `training.train_one_fold`.

Modelo completo (LM + predictor): `models.ModelConfig`, `models.build_model`, `models.TenderSuccessModel`, `models.ChunkEmbedder`. Requiere `transformers` (incluido en dependencias).

### Extracción PDF → Markdown (`doc_extract`)

`doc_extract.PDFReader`: pdfplumber (texto) + camelot lattice/stream (tablas), mismo flujo que tu script.

```bash
uv sync --extra pdf
```

**Sistema:** Camelot *lattice* suele requerir [Ghostscript](https://www.ghostscript.com/) instalado y en el `PATH`. Sin eso, esas lecturas pueden fallar con warning (stream a veces es más tolerante).

Uso: `from doc_extract import PDFReader` → `PDFReader(ruta.pdf).read_pdf_as_markdown()`.

## Configuración

- **Secretos y Spaces:** `.env` (plantilla: `.env.example`).
- **Rutas y nombres de variables:** `configs/data.yaml`.

El código de entrenamiento o ETL lo escribís vos en `src/` u otra carpeta; este repo solo fija proyecto `uv`, ignorados y convención de config.

## Materializar `procurements.json` (una vez)

Lee `tagging/ids/ids.json` y escribe `outcome-predictor/procurements/procurements.json` con `"status": "complete"` en cada elemento (no toca el archivo fuente).

```bash
uv run python scripts/once/materialize_procurements_with_status.py --dry-run
uv run python scripts/once/materialize_procurements_with_status.py
```

Si usás `SPACES_PREFIX` en `.env`, las keys anteriores se anteponen a ese prefijo.

## ETL: más IDs por estado (`unsuccessful`, `cancelled`)

Descarga desde la API DNCP (misma familia de filtros que `complete`) y sube **un archivo por estado** en la **misma carpeta que `procurements.json`**: `outcome-predictor/procurements/` respecto del bucket, anteponiendo `SPACES_PREFIX` si lo definís (misma regla que el script materialize). Cada registro incluye `"status"`.

- `ids_unsuccessful.json` + `checkpoint_unsuccessful.json`
- `ids_cancelled.json` + `checkpoint_cancelled.json`

Override de carpeta: `DO_SPACES_DATASET_PREFIX` o `--prefix` (ruta completa bajo el bucket).

Objetivo por defecto: **1000** IDs únicos por estado (deduplicado por `tenderId`). Ajustá con `--min`.

```bash
uv run python scripts/etl/fetch_ids_by_status.py unsuccessful
uv run python scripts/etl/fetch_ids_by_status.py cancelled
uv run python scripts/etl/fetch_ids_by_status.py both
```

Credenciales Spaces: sirven `SPACES_*` del `.env.example` o los alias `DO_SPACES_*` / `AWS_*` descritos en el docstring del script.

## Dataset unificado + pliegos PDF (PBC)

Fusiona los tres JSON del prefijo del dataset (**orden fijo de lectura y de cola de descarga:** `ids_unsuccessful.json` → `ids_cancelled.json` → `procurements.json`, para priorizar PBCs de no exitosas al cortar y re-ejecutar). Intenta bajar **solo PDF** de pliego / carta de invitación; si no hay candidato en PDF, deja `pbc_downloaded: false` y `pbc_skip_reason`.

- **PDFs en Spaces** (reanudable con `head_object`): `outcome-predictor/pbcs/pdf/{tenderId_sanitizado}.pdf` respecto del bucket + `SPACES_PREFIX` si lo usás. Cada fila con éxito incluye `pbc_s3_key`.
- JSON enriquecido: `data/processed/procurements_dataset.json` y subida a `…/procurements_dataset.json` en Spaces (salvo `--no-upload`).

```bash
uv sync
uv run python scripts/etl/merge_and_download_pbcs.py --dry-run
uv run python scripts/etl/merge_and_download_pbcs.py
uv run python scripts/etl/merge_and_download_pbcs.py --limit 30   # prueba con las primeras 30 licitaciones
```
