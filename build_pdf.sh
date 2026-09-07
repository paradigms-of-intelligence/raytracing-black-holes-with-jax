#!/bin/sh

echo "Rebuilding 'extra material' PDFs from Markdown..."
pandoc -s -f markdown -t pdf coordinate_transformations/ct_howto.md \
       -o coordinate_transformations/ct_howto.pdf

echo "Rebuilding main article from TeX..."
(cd tex; pdflatex main.tex; bibtex main; pdflatex main.tex; rm -f main.{aux,bbl,blg,log,out})
