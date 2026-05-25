#!/bin/bash
# Build script for ACL/EMNLP paper
# Usage: ./compile.sh

set -e  # exit on first error
cd "$(dirname "$0")"

PAPER=main

# Clean stale build artifacts
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.fls *.fdb_latexmk sections/*.aux

# Build with bibtex, then re-run pdflatex until lineno/labels converge.
# lineno's switch mode (per-column outer-margin numbering) needs the .aux
# to stabilize; otherwise numbers get assigned to the wrong column and
# visually overlap with the opposite column's text.
pdflatex -interaction=nonstopmode "$PAPER.tex"
bibtex "$PAPER"
for i in 1 2 3 4 5; do
    pdflatex -interaction=nonstopmode "$PAPER.tex"
    if ! iconv -f ISO-8859-1 -t UTF-8 "$PAPER.log" 2>/dev/null \
         | grep -qE "Linenumber reference failed|Label\(s\) may have changed"; then
        echo "Converged after pass $i."
        break
    fi
done

# Report status
if [ -f "$PAPER.pdf" ]; then
    echo ""
    echo "=== BUILD OK ==="
    echo "Output: $(pwd)/$PAPER.pdf"
    echo "Pages:  $(pdfinfo $PAPER.pdf | grep ^Pages | awk '{print $2}')"
    echo "Size:   $(du -h $PAPER.pdf | awk '{print $1}')"
    echo ""
    echo "=== UNRESOLVED REFS/CITES (should be empty) ==="
    grep -iE "undefined|multiply defined|warning.*citation" "$PAPER.log" | head -10 || echo "(none)"
else
    echo ""
    echo "=== BUILD FAILED ==="
    echo "Last errors:"
    grep -A 2 "^!" "$PAPER.log" | head -20
    exit 1
fi