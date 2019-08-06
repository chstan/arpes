+-----------------------+
| **Documentation**     |
+=======================+
| |Documentation|       |
+-----------------------+

.. |Documentation| image:: https://img.shields.io/badge/api-reference-blue.svg
   :target: https://arpes.netlify.com/

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

PyARPES
=======

PyARPES simplifies the analysis and collection of angle-resolved photoemission spectroscopy (ARPES) and emphasizes

* modern, best practices for data science
* support for a standard library of ARPES analysis tools mirroring those available in Igor Pro
* interactive and extensible analysis tools

It supports a variety of data formats from synchrotron and laser-ARPES sources including ARPES at the Advanced
Light Source (ALS), the data produced by Scienta Omicron GmbH's "SES Wrapper", data and experiment files from
Igor Pro, NeXuS files, and others.

To learn more about installing and using PyARPES in your analysis or data collection application,
visit `the documentation site`_.

PyARPES is currently developed by Conrad Stansbury of the Lanzara Group at the University of California, Berkeley.

Installation
============

PyARPES can be installed from source, or using either ``pip`` or ``conda`` into a Python 3.6 or 3.7 environment.
``conda`` is preferred as a package manager in order to facilitate installing the libraries for reading HDF and
NetCDF files.

Pip installation
----------------

::

   pip install arpes

Platform specific instructions to install the HDF and NetCDF libraries are
available below.

Conda installation
------------------

PyARPES is distributed through the ``arpes`` Anaconda channel, but includes dependencies through ``conda-forge``.
A minimal install looks like

::

   conda install -c arpes -c conda-forge arpes


Local installation from source
------------------------------

If you want to modify the source for PyARPES as you use it, you might prefer a local installation from source.
Details can be found on `the documentation site`_.


Suggested steps
---------------

1. Clone or duplicate the folder structure in the repository ``arpes-analysis-scaffold``,
   skipping the example folder and data if you like
2. Install and configure standard tools like Jupyter_ or Jupyter Lab. Notes on installing
   and configuring Jupyter based installations can be found in ``jupyter.md``
3. Explore the documentation and example notebooks at `the documentation site`_.

Contact
=======

Questions, difficulties, and suggestions can be directed to Conrad Stansbury (chstan@berkeley.edu)
or added to the repository as an issue. In the case of trouble, also check the `FAQ`_.

Copyright |copy| 2018-2019 by Conrad Stansbury, all rights reserved.

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN

.. _Jupyter: https://jupyter.org/
.. _the documentation site: https://arpes.netlify.com/
.. _contributing: https://arpes.netlify.com/#/contributing
.. _FAQ: https://arpes.netlify.com/#/faq

