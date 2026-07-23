# Libro ampliado derivado del shortpaper

El documento principal es `main.tex`. El libro mantiene el contenido traducido
del shortpaper en `Capitulos/shortpaper_fragments/` y lo amplía en seis
capítulos:

- `p0010Introduccion.tex`;
- `p0020MTeorico.tex`;
- `p0030TrabajosRelacionados.tex` (incluye el estado del arte);
- `p0040PropuestaExperimento.tex`;
- `p0050Resultados.tex`;
- `p0060Conclusion.tex`.

La carpeta de figuras se sincroniza desde `../paper/` ejecutando:

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

La bibliografía ampliada, el cuerpo traducido, el resumen en español y la lista
de acrónimos se mantienen manualmente. El script no sobrescribe
`references.bib` cuando ese archivo ya existe.
