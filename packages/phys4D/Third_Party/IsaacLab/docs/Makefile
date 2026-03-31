




SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: multi-docs
multi-docs:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@cp _redirect/index.html $(BUILDDIR)/index.html

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)
