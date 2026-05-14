Camera-ready package contents

This package is intended for EasyChair re-upload as a zip file.

A copy of the camera-ready tree (LaTeX + PDF + figures + peer-review export),
ready to add payment and then zip, lives in:

  paper/submission/camera_ready_ICEDEG/

See `README_BEFORE_ZIP.txt` inside that folder.

Included:
- shortpaper.tex
- shortpaper.bib (references; BibTeX + IEEEtran.bst)
- shortpaper.bbl (generated; regenerate with BibTeX after editing .bib)
- shortpaper.pdf
- IEEEtran.cls
- IEEEtran.bst (copy of vendor/IEEEtranBST2/IEEEtran.bst for local BibTeX)
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

The canonical BibTeX style bundle is under `paper/vendor/IEEEtranBST2/`;
`paper/IEEEtran.bst` is a copy so BibTeX finds the style without extra paths.
