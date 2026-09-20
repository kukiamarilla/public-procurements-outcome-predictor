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
