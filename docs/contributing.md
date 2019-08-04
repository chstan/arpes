# Contributing

## Why

Please **do** add useful new analysis routines here. Having a central source of analysis code across the group will make it easier to work across different environments, and allow us all to access cool new analysis techniques.

## How to Contribute

1. You will need a git client, if you don't want to use a terminal, have a look at Github's [GUI Client](https://desktop.github.com/)
2. Write your new analysis code
3. Put it someplace reasonable in line with the project's organizational principles
4. Add convenience accessors on `.T`, `.S`, or `.F` as relelvant
5. If necessary, add imports to `01-common-imports.ipy` or to `setup` in `arpes.__init__.py`
6. Make sure the new code is adequately documented with a 
   [docstring](https://en.wikipedia.org/wiki/Docstring#Python).
7. Add documentation to this documentation site if relevant, see below for details  
8. If you added new requirements, make sure they get added to `requirements.txt`/`setup.py`
9. Ensure you have the latest code by `git pull`ing as necessary, to prevent any conflicts
10. `git commit` your change and `git push` (do not ever `git push --force`)

## Notes on Authorship

Some sense of authorship of various parts of the analysis code are automatically retained. `git` remembers every change in the project and who contributed them. You can view this at the terminal with `git blame`, or from the web or Desktop clients.
n
If you use (or contribute) a large amount of new work that ends up being a substantial part of someone elses' analysis effort, please consider citing (requesting citation) or including the author (requesting authorship) of the analysis code you used as a coauthor, as you would in other situations.

If you would like or feel that it would be beneficial as far as users seeking help, you can also put authorship information into the docstring, but note that the docstring should **not** become a record of changes to a piece of analysis code.

At some point in the near future, Conrad will put together a preprint to be placed on the arXiv with some 
information on this library, its core features, and how it should be used. Clients of the analysis code 
should cite this preprint once it becomes available.

## Contributing Documentation

### Adding a new section

To add a new section to the documentation, modify `_sidebar_partial.md` and add any other 
associated files. Then follow the instructions below for rebuilding the documentation.

### Updating existing documentation

To update existing documentation you can simply modify the appropriate files. 
You should not need to rebuild the documentation for your changes to take effect, but there
is no harm is doing so.

### Rebuilding the documentation

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