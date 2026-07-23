# Libro derivado del shortpaper

El documento principal es `main.tex`. El cuerpo del libro está traducido al
español en `Capitulos/p0010contenido.tex`. La bibliografía y la carpeta de
figuras se sincronizan desde `../paper/` ejecutando:

```bash
python3 generar_desde_paper.py
```

Para compilar el libro:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

El cuerpo traducido, el resumen en español y la lista de acrónimos se mantienen
manualmente en la carpeta `Capitulos/` para evitar que la sincronización vuelva
a introducir el texto inglés del shortpaper.
