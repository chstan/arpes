Developer Guide
===============

Topics
------

Installing an editable copy of PyARPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Make an Anaconda environment or ``virtualenv`` for your installation
   of PyARPES
2. Clone the respository

.. code:: bash

   git clone https://gitlab.com/lanzara-group/python-arpes

or

.. code:: bash

   git clone https://github.com/chstan/arpes

3. Install PyARPES into your conda environment/virtualenv with
   ``pip install -e .``

Tests
~~~~~

Prerequisites
^^^^^^^^^^^^^

You need some additional Python packages as well as Yarn.

Installing Test Requirements
''''''''''''''''''''''''''''

Install additional test requirements by running

.. code:: bash

   $> conda env update --file environment-update-test.yml

or by manually installing the requirements.

Installing Yarn
'''''''''''''''

Follow instructions for your platform at
`yarnpkg.com <https://yarnpkg.com/>`__.

Running Tests
^^^^^^^^^^^^^

.. code:: bash

   yarn test

Getting test coverage information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to generate coverage information, you can run the coverage
command and serve results as HTML locally.

In one terminal start the Python HTTP server with:

.. code:: bash

   $> python -m http.server --directory htmlcov

Now, refresh coverage information with

.. code:: bash

   yarn coverage

finally, you can view results at localhost:8000.

When to write tests
^^^^^^^^^^^^^^^^^^^

If you are adding a new feature, please consider adding a few unit
tests. Additionally, all bug fixes should come with a regression test if
they do not require a very heavy piece of fixture data to support them.

To write a test that consumes data from disk using the standard PyARPES
loading conventions, fixtures are available in ``tests/conftest.py``.
The tests extent in ``test_basic_data_loading.py`` illustrate using
these fixtures.

Contributing Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new section
^^^^^^^^^^^^^^^^^^^^

To add a new section to the documentation, modify
``_sidebar_partial.md`` and add any other associated files. Then follow
the instructions below for rebuilding the documentation.

Updating existing documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update existing documentation you can simply modify the appropriate
files. You should not need to rebuild the documentation for your changes
to take effect, but there is no harm is doing so.

Rebuilding the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To rebuild the documentation you will need to have both
`sphinx <http://www.sphinx-doc.org/en/master/>`__ and
`pandoc <https://pandoc.org/>`__ installed. Then from the directory that
contains the ``setup.py`` file

1. Refresh Sphinx sources with ``sphinx-apidoc``:
   ``python -m sphinx.apidoc --separate -d 3 --tocfile toc -o source arpes --force``
2. Build Sphinx documentation to ReStructuredText:
   ``make clean && make rst``
3. Convert ReStructuredText to Markdown: ``./source/pandoc_convert.py``
4. Run ``docsify`` to verify changes: ``docsify serve ./docs``
5. As desired publish to docs site by pushing updated documentation

**Note** Sometimes ``sphinx-doc`` has trouble converting modules to
ReStructured Text.versioning This typically manifests with a
``KeyError`` in ``docutils``. This occurs when the docstrings do not
conform to the standard for ReStructuredText. The most common problem
encountered is due to bare hyperlinks, which are incompatible with the
*unique* hyperlink format in RST.

Style
~~~~~

We don’t have any hard and fast style rules. As a coarse rule of thumb,
if your code scans well and doesn’t use too many short variable names
there’s no issue.
