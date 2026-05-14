Camera-ready package contents

This package is intended for EasyChair re-upload as a zip file.

Included:
- shortpaper.tex
- shortpaper.bib (references; BibTeX + IEEEtran.bst)
- shortpaper.bbl (generated; regenerate with BibTeX after editing .bib)
- shortpaper.pdf
- IEEEtran.cls
- IEEEtran.bst (copy of IEEEtranBST2/IEEEtran.bst for local BibTeX)
- figures/pr_curve_cv.png
- figures/calibration_curve_cv.png
- figures/threshold_sensitivity_comparison.png

Required by the conference but not found automatically in this repository:
- point-by-point comments review
- payment confirmation

If you have those files, add them to the zip before upload or let the assistant
package them for you after placing them in the workspace.

Build (from this directory):
  pdflatex shortpaper
  bibtex shortpaper
  pdflatex shortpaper
  pdflatex shortpaper

The canonical BibTeX style in the repository is IEEEtranBST2/IEEEtran.bst;
paper/IEEEtran.bst is a copy so BibTeX finds the style without extra paths.
