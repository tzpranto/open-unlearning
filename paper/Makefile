MAIN  = main
LATEX = pdflatex
BIB   = bibtex

.PHONY: all clean watch

all: $(MAIN).pdf

SECTIONS = $(wildcard sections/*.tex)

$(MAIN).pdf: $(MAIN).tex macros.tex references.bib neurips_2025.sty $(SECTIONS)
	$(LATEX) $(MAIN)
	-$(BIB) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

quick:
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).{pdf,aux,bbl,blg,log,out,toc,nav,snm,fls,fdb_latexmk,synctex.gz}
	rm -f sections/*.aux

watch:
	@echo "Watching for changes... (requires inotifywait)"
	@while true; do \
		inotifywait -qe modify $(MAIN).tex macros.tex references.bib sections/*.tex; \
		make all; \
	done
