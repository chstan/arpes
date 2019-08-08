# Developer Guide

## Topics

### Installing an editable copy of PyARPES

1. Make an Anaconda environment or `virtualenv` for your installation of PyARPES
2. Clone the respository

```bash
git clone https://gitlab.com/lanzara-group/python-arpes
```

or 

```bash
git clone https://github.com/chstan/arpes
```

3. Install PyARPES into your conda environment/virtualenv with `pip install -e .`


### Tests

#### Running tests

You should check that tests pass before merging code in to PyARPES. Sometimes things work 
locally and break the build anyway: we are still improving our infrastructure to make sure
the Azure Pipelines environment is realistic.

In order to run the tests, you'll need to install `pytest`. Otherwise, there shouldn't be
any other dependencies. From the root of the repository, you can then run the full test suite with

```bash
pytest
```

If you want to generate coverage information, you can use command that runs on
the CI server with

```bash
pytest tests --doctest-modules --junitxml=junit/test-results.xml --cov=arpes --cov-report=xml --cov-report=html
``` 

#### When to write tests

If you are adding a new feature, please consider adding a few unit tests. Additionally, all bug fixes 
should come with a regression test if they do not require a very heavy piece of fixture data to support them.

To write a test that consumes data from disk using the standard PyARPES loading conventions, 
fixtures are available in `tests/conftest.py`. The tests extent in
`test_basic_data_loading.py` illustrate using these fixtures.

### Contributing Documentation

#### Adding a new section

To add a new section to the documentation, modify `_sidebar_partial.md` and add any other 
associated files. Then follow the instructions below for rebuilding the documentation.

#### Updating existing documentation

To update existing documentation you can simply modify the appropriate files. 
You should not need to rebuild the documentation for your changes to take effect, but there
is no harm is doing so.

#### Rebuilding the documentation

To rebuild the documentation you will need to have both [sphinx](http://www.sphinx-doc.org/en/master/) 
and [pandoc](https://pandoc.org/) installed. Then from the directory that contains the `setup.py` file 

1. Refresh Sphinx sources with ``sphinx-apidoc``:
   ``python -m sphinx.apidoc --separate -d 3 --tocfile toc -o source arpes --force``
2. Build Sphinx documentation to ReStructuredText:
   ``make clean && make rst``
3. Convert ReStructuredText to Markdown: ``./source/pandoc_convert.py``
4. Run ``docsify`` to verify changes: ``docsify serve ./docs``
5. As desired publish to docs site by pushing updated documentation

**Note** Sometimes `sphinx-doc` has trouble converting modules to ReStructured Text.
This typically manifests with a `KeyError` in `docutils`. This occurs when the docstrings
do not conform to the standard for ReStructuredText. The most common problem encountered is due to 
bare hyperlinks, which are incompatible with the *unique* hyperlink format in RST. 

### Style

We don't have any hard and fast style rules. As a coarse rule of thumb,
if your code scans well and doesn't use too many short variable names
there's no issue.    