# Paper (ICEDEG / IEEE)

Layout under `paper/`:

| Path | Role |
|------|------|
| `shortpaper.tex`, `shortpaper.pdf`, `shortpaper.bib`, `shortpaper.bbl` | Main manuscript and bibliography. |
| `IEEEtran.cls`, `IEEEtran.bst` | Class and BibTeX style (copies for a self-contained build in this folder). |
| `figures/` | PNG figures included by the manuscript. |
| `scripts/` | Optional helpers to regenerate plots or tables. |
| `vendor/` | IEEE templates and BibTeX reference bundles (read-only reference; see `vendor/README.md`). |
| `submission/camera_ready_ICEDEG/` | EasyChair camera-ready bundle (sources + figures + review export); add payment, then zip. |
| `peer-review.txt` | EasyChair reviewer export (reference). |
| `camera_ready_README.txt` | Checklist and build commands for the camera-ready package. |

**Build** (from `paper/`):

```bash
pdflatex shortpaper
bibtex shortpaper
pdflatex shortpaper
pdflatex shortpaper
```
