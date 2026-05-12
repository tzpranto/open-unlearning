#!/bin/bash
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex | tail -3
