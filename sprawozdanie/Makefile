PANDOC = pandoc
PANDOC_FLAGS = -V lang=pl -F pandoc-crossref
LATEX_FLAGS = --pdf-engine=lualatex
TARGETS = sprawozdanie.pdf
COMMON_DEPS = Makefile $(wildcard img/*.*)

all: $(TARGETS)

.PHONY: clean
clean:
	-rm -f $(TARGETS)

%.pdf: %.md $(COMMON_DEPS)
	$(PANDOC) $(PANDOC_FLAGS) $(LATEX_FLAGS) $< -o $@
