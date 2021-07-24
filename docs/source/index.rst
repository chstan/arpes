PyARPES
=======

**December 2020, V3 Release**: The current relase focuses on improving
usage and workflow for less experienced Python users, lifting version
incompatibilities with dependencies, and ironing out edges in the user
experience.

For the most part, existing users of PyARPES should have no issues
upgrading, but we now require Python 3.8 instead of 3.7. We now provide
a conda environment specification which makes this process simpler, see
the installation notes below. It is recommended that you make a new
environment when you upgrade.

.. raw:: html

   <figure>
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Gd0qJuInzvE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
     </iframe>
     <figcaption>
       You can find more usage and example videos here.
     </figcaption>
   </figure>

PyARPES is an open-source data analysis library for angle-resolved
photoemission spectroscopic (ARPES) research and tool development. While
the scope of what can be achieved with PyARPES is general, PyARPES
focuses on creating a productive programming and analysis environment
for ARPES and its derivatives (Spin-ARPES, ultrafast/Tr-ARPES,
ARPE-microspectroscopy, etc).

As part of this mission, PyARPES aims to reduce the feedback cycle for
scientists between data collection and producing publication quality
analyses and figures. Additionally, PyARPES aims to be a platform on
which new types of ARPES and spectroscopic analyses can be rapidly
prototyped and tested.

For these reasons, PyARPES includes out of the box a **large variety of
analysis tools** for

1.  Applying corrections to ARPES data
2.  Doing gap analysis
3.  Performing sophisticated band analysis
4.  Performing rapid and automated curve fitting, even over several
    dataset dimensions
5.  Background subtraction
6.  Dataset collation and combination
7.  Producing common types of ARPES figures and reference figures
8.  Converting to momentum space
9.  Interactively masking, selecting, laying fits, and exploring data
10. Plotting data onto Brillouin zones

These are in addition to facilities for derivatives, symmetrization, gap
fitting, Fermi-Dirac normalization, the minimum gradient method, and
others. Have a look through the `crash course </how-to>`__ to learn
about supported workflows.

By default, PyARPES supports a variety of data formats from synchrotron
and laser-ARPES sources including ARPES at the Advanced Light Source
(ALS), the data produced by Scienta Omicron GmbH’s “SES Wrapper”, data
and experiment files from Igor Pro (see in particular the section on
`importing Igor Data </igor-pro>`__), NeXuS files, and others.
Additional data formats can be added via a user plugin system.

If PyARPES helps you in preparing a conference presentation or
publication, please respect the guidelines for citation laid out in the
notes on `user contribution </contributing>`__. Contributions and
suggestions from the community are also welcomed warmly.

Tool Development
^^^^^^^^^^^^^^^^

Secondary to providing a healthy and sane analysis environment, PyARPES
is a testbed for new analysis and correction techniques, and as such
ships with ``scikit-learn`` and ``open-cv`` as compatible dependencies
for machine learning. ``cvxpy`` can also be included for convex
optimization tools.

Installation
============

See the :doc:`installation` page for details.

Contributing and Documentation
==============================

See the section on the docs site about
`contributing <https://arpes.readthedocs.io/contributing>`__ for
information on adding to PyARPES and rebuilding documentation from
source.

Copyright © 2018-2020 by Conrad Stansbury, all rights reserved. Logo
design, Michael Khachatrian

**Installation + Technical Notes**

* :doc:`installation`
* :doc:`migration-guide`
* :doc:`faq`
* :doc:`example-videos`
* :doc:`api`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Installation + Technical Notes

   installation
   faq
   example-videos
   api

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorial

   Example Data <notebooks/tutorial-data>
   Jupyter Crash Course <notebooks/jupyter-crash-course>
   Data Exploration <notebooks/basic-data-exploration>
   Data Manipulation <notebooks/data-manipulation-intermediate>
   `xarray` Extensions Pt. 1 <notebooks/custom-dot-s-functionality>
   `xarray` Extensions Pt. 2 <notebooks/custom-dot-t-functionality>
   Curve Fitting <notebooks/curve-fitting>
   Fermi Edge Corrections <notebooks/fermi-edge-correction>
   Momentum Conversion <notebooks/converting-to-kspace>
   Nano XPS Analysis <notebooks/full-analysis-xps>

**Detailed Guides**

* :doc:`loading-data`
* :doc:`interactive`
* :doc:`workspaces`
* :doc:`statistics`
* :doc:`curve-fitting`
* :doc:`customization`
* :doc:`advanced-plotting`
* :doc:`writing-plugins-basic`
* :doc:`writing-plugins`
* :doc:`igor-pro`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Detailed Guides

   loading-data
   interactive
   workspaces
   statistics
   curve-fitting
   customization
   advanced-plotting
   writing-plugins-basic
   writing-plugins
   igor-pro

**ARPES**

* :doc:`spectra`
* :doc:`momentum-conversion`
* :doc:`th-arpes`
* :doc:`single-particle-spectral`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: ARPES

   spectra
   momentum-conversion
   th-arpes
   single-particle-spectral

**Plotting**

* :doc:`stack-plots`
* :doc:`brillouin-zones`
* :doc:`fermi-surfaces`
* :doc:`3d-cut-plots`
* :doc:`spin-arpes`
* :doc:`tr-arpes`
* :doc:`annotations`
* :doc:`plotting-utilities`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Plotting

   stack-plots
   brillouin-zones
   fermi-surfaces
   3d-cut-plots
   spin-arpes
   tr-arpes
   annotations
   plotting-utilities

**Reference**

* :doc:`migration-guide`
* :doc:`writing-interactive-tools`
* :doc:`writing-plugins-basic`
* :doc:`writing-plugins`
* :doc:`data-provenance`
* :doc:`modeling`
* :doc:`cmp-stack`
* :doc:`contributing`
* :doc:`dev-guide`
* :doc:`api`
* :doc:`CHANGELOG`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   migration-guide
   writing-interactive-tools
   writing-plugins-basic
   writing-plugins
   data-provenance
   modeling
   cmp-stack
   contributing
   dev-guide
   api
   CHANGELOG
   