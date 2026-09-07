#!/bin/sh

echo "Rebuilding PDFs from Markdown..."
pandoc -s -f markdown -t pdf coordinate_transformations/ct_howto.md \
       -o coordinate_transformations/ct_howto.pdf
