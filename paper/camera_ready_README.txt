<<<<<<< HEAD
Camera-ready package contents

This package is intended for EasyChair re-upload as a zip file.

Included:
- shortpaper.tex
- shortpaper.pdf
- IEEEtran.cls
- figures/pr_curve_cv.png
- figures/calibration_curve_cv.png
- figures/threshold_sensitivity_comparison.png

Required by the conference but not found automatically in this repository:
- point-by-point comments review
- payment confirmation

If you have those files, add them to the zip before upload or let the assistant
package them for you after placing them in the workspace.
=======
Camera-ready build (IEEE / ICEDEG)

Build the manuscript from this directory (`paper/`):

  pdflatex shortpaper
  bibtex shortpaper
  pdflatex shortpaper
  pdflatex shortpaper

Figures are under `paper/figures/`. The canonical BibTeX style bundle is under
`paper/vendor/IEEEtranBST2/`; `paper/IEEEtran.bst` is a copy for local BibTeX.

EasyChair upload packages (zip, payment confirmation, reviewer response letter,
peer-review exports) are kept outside version control. See `.gitignore`.
>>>>>>> paper-camera-ready-from-submission
