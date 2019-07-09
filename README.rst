|Banner|

.. |Banner| image:: PyPES-Banner.png
            :alt: PyPES Banner

+-----------------------+
| **Documentation**     |
+=======================+
| |Documentation|       |
+-----------------------+

.. |Documentation| image:: https://img.shields.io/badge/api-reference-blue.svg
   :target: https://stupefied-bhabha-ce8a9f.netlify.com/

.. image:: https://dev.azure.com/lanzara-group/PyARPES/_apis/build/status/PyARPES%20CI%20Build?branchName=master
   :target: https://dev.azure.com/lanzara-group/PyARPES/_build?definitionId=2

.. image:: https://img.shields.io/azure-devops/coverage/lanzara-group/PyARPES/2.svg
   :target: https://dev.azure.com/lanzara-group/PyARPES/_build?definitionId=2

.. image:: https://img.shields.io/pypi/v/arpes.svg
   :target: https://pypi.org/project/arpes/

.. image:: https://img.shields.io/conda/v/arpes/arpes.svg
   :target: https://anaconda.org/arpes/arpes

.. image:: https://img.shields.io/pypi/pyversions/arpes.svg
   :target: https://pypi.org/project/arpes/



Installation
============

The simplest way to install the package is using pip. While this repository
is private, you can install with pip by pointing pip to the URL of the repository
as:

::

   pip install --process-dependency-links -e


Once this project is published on PyPI, you can install by using

::

   pip install arpes


You will need to install into a Python interpreter with version 3.5 or higher. Note that the
`--process-dependency-links` directive appears in both commands and is necessary in order to
include a patched version of the Igor interop library. If you have a newer version of ``pip``,
you might not be able to use `-process-dependency-links` as the ``pip`` team has deprecated
this option. Please consult the `FAQ`_ for how to manually install a few extra dependencies if this
is the case.

Windows Installation
--------------------

Windows is not the most friendly operating system for scientific software. Although
installation is absolutely possible manually, I cannot advocate Anaconda enough for Windows
users, as it smooths out a lot of inhomogeneity in the process. For details on how to
install manually, consult also the `FAQ`_.


Alternative Installation
========================

If for whatever reason you do not want to install the project as a package but would
like to import it locally, legacy install instructions are available in the first section
of ``README.legacy.rst``.

This can be advantageous if you want to frequently change the package source without
reinstalling. A further alternative is to clone the package and ``pip install`` from the local folder,
which has the advantage of a simple installation procedure and puts the code someplace easily editable.


Additional Suggested Steps
--------------------------

1. Clone or duplicate the folder structure in the repository ``arpes-analysis-scaffold``,
   skipping the example folder and data if you like
2. Install and configure standard tools like Jupyter_ or Jupyter Lab. Notes on installing
   and configuring Jupyter based installations can be found in ``jupyter.md``
3. Explore the documentation and example notebooks at `the documentation site`_.

``local_config.py``
-------------------

The local configuration allows you to override the settings that are
committed to the repository and therefore shared. You can use this to
as adjust settings on various interactive tools. For reference, Conrad’s
looks like:

::

   SETTINGS = {
       'interactive': {
           'main_width': 600,
           'marginal_width': 300,
           'palette': 'magma',
       },
   }

IPython Kernel Customization
----------------------------

If you don’t want to have to import everything all the time, you should
customize your IPython session so that it runs imports when you first
spin up a kernel. There are good directions for how to do this online,
but a short version is:

1. Create an IPython profile, use this to start your notebooks
2. In ``~/.ipython/profile_default/`` make a folder ``startup``
3. Add the files
   ``~/.ipython/profile_default/startup/00-add-arpes-path.py`` and
   ``~/.ipython/{Your profile}/startup/01-common-imports.ipy`` according
   to the templates in ``ipython_templates``. See in particular note
   above about setting the environment variable using this file.
4. Customize

It is important that the filenames you put are such that
``00-add-arpes-path`` is lexographically first, as this ensures that it is
executed first. The ``.ipy`` extension on ``01-common-imports.ipy`` is
also essential. Ask Conrad if any of this is confusing.

Contributing and Documentation
==============================

See the section on the docs site about `contributing`_ for information on
adding to PyPES and rebuilding documentation from source.

Contact
=======

Questions, difficulties, and suggestions can be directed to Conrad Stansbury (chstan@berkeley.edu)
or added to the repository as an issue. In the case trouble, also check the `FAQ`_.

Copyright |copy| 2018 by Conrad Stansbury, all rights reserved.

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN

.. _Jupyter: https://jupyter.org/
.. _the documentation site: https://pypes.netlify.com/
.. _contributing: https://pypes.netlify.com/#/contributing
.. _FAQ: https://pypes.netlify.com/#/faq

