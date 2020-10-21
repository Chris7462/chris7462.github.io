LATEX=/usr/bin/latex
DVIPDFMX=/usr/bin/dvipdfmx
CV=CoverLetter

defalut:
	${LATEX} ${CV}.tex
	${LATEX} ${CV}.tex
	${DVIPDFMX} ${CV}.dvi

clean:
	rm -f *.bib *.out *.xml *.ps *.log *.dvi *.aux *.cjk *.lof *.toc *.lot *.blg *.bbl *~ #*.pdf ;cd ${MPDIR};make clean
