# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -m sphinx
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile rst2markdown clean-pypi dist-pypi clean-build-pypi dist-pypi bdist dist-pypi-sdist dist-pypi upload-pypi clean-conda dist-conda upload-conda install-conda-local clean dist upload install test tools-update sanity-check dist-pypi

rst2markdown:
	echo "test"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# From here, much of our Makefile infrastructure is modified from Fast.ai

version = $(shell python setup.py --version)
branch = $(shell git branch | grep \* | cut -d ' ' -f2)

clean-pypi: clean-build-pypi clean-pyc-pypi

dist-pypi: | clean-pypi dist-pypi-sdist dist-pypi-bdist
	ls -l dist

clean-build-pypi: ## remove pypi build artifacts
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc-pypi: ## remove python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

dist-pypi-bdist: ## build pypi wheel package
	@echo "\n\n*** Building pypi wheel package"
	python setup.py bdist_wheel

dist-pypi-sdist: ## build pypi source package
	@echo "\n\n*** Building pypi source package"
	python setup.py sdist

dist-pypi: | clean-pypi dist-pypi-sdist dist-pypi-bdist ## build pypi source and wheel package
	ls -l dist

upload-pypi: ## upload pypi package
	@echo "\n\n*** Uploading" dist/* "to pypi\n"
	twine upload dist/*


## Conda Related
clean-conda:
	@echo "\n\n*** conda build purge"
	conda build purge-all
	@echo "\n\n*** rm -rf conda-dist/"
	rm -rf conda-dist/

dist-conda: | clean-conda dist-pypi-sdist
	@echo "\n\n*** Building conda package"
	mkdir "conda-dist"
	conda-build ./conda/ --output-folder conda-dist # CHECK ME
	ls -l conda-dist/noarch/*tar.bz2

upload-conda:
	@echo "\n\n*** Uploading" conda-dist/noarch/*tar.bz2 "to arpes@anaconda.org\n"
	anaconda upload conda-dist/noarch/*tar.bz2 -u arpes

install-conda-local:
	@echo "\n\n*** Installing the local build of" conda-dist/noarch/*tar.bz2
	conda install -y -c ./conda-dist/ -c arpes arpes==$(version)


### Combined pip and conda builds
clean: clean-pypi clean-conda

dist: clean dist-pypi dist-conda

upload: upload-pypi upload-conda

install: clean
	python setup.py install

test:
	pytest --runslow -ra

tools-update:
	conda install -y conda-verify conda-build anaconda-client
	pip install -U twine

log_file := log/release-`date +"%Y-%m-%d-%H-%M-%S"`.log

sanity-check:
	@echo "\n\n*** Checking branch before release: should be: release-X.Y.Z"
	@perl -le '$$_=shift; $$v="branch name: $$_"; /release-\.$$/ ? print "Good $$v" : die "Bad $$v, expecting release-"' $(branch)
