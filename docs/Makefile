# Minimal makefile for Sphinx documentation

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help
help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

%:
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"
