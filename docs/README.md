# PyARPES

PyARPES is an open-source data analysis library for angle-resolved photoemission spectroscopic (ARPES) research 
and tool development. While the scope of what can be achieved with PyARPES is general, PyARPES focuses on creating a 
productive programming and analysis environment for ARPES and its derivatives (Spin-ARPES, ultrafast/Tr-ARPES, 
ARPE-microspectroscopy, etc).

As part of this mission, PyARPES aims to reduce the feedback cycle for scientists between data collection and 
producing publication quality analyses and figures. Additionally, PyARPES aims to be a platform on which new types 
of ARPES and spectroscopic analyses can be rapidly prototyped and tested.

For these reasons, PyARPES includes out of the box a **large variety of analysis tools** for

1. Applying corrections to ARPES data
2. Doing gap analysis
3. Performing sophisticated band analysis
4. Performing rapid and automated curve fitting, even over several dataset dimensions
5. Background subtraction
6. Dataset collation and combination
7. Producing common types of ARPES figures and reference figures
8. Converting to momentum space
9. Interactively masking, selecting, laying fits, and exploring data
10. Plotting data onto Brillouin zones

These are in addition to facilities for derivatives, symmetrization, gap fitting, 
Fermi-Dirac normalization, the minimum gradient method, and others. Have a look
through the [crash course](/how-to) to learn about supported workflows.  

By default, PyARPES supports a variety of data formats from synchrotron and laser-ARPES sources including ARPES at 
the Advanced Light Source (ALS), the data produced by Scienta Omicron GmbH's "SES Wrapper", data and experiment 
files from Igor Pro, NeXuS files, and others. Additional data formats can be added via a user plugin system.

If PyARPES helps you in preparing a conference presentation or publication, please respect the guidelines 
for citation laid out in the notes on [user contribution](/contributing). Contributions and suggestions from the 
community are also welcomed warmly.

#### Tool Development

Secondary to providing a healthy and sane analysis environment, PyARPES is a testbed for new analysis and 
correction techniques, and as such ships with `scikit-learn` and `open-cv` as compatible dependencies for 
machine learning. `cvxpy` can also be included for convex optimization tools.


## Installation

Please check the package and platform requirements before installing PyARPES. For some technical reasons, PyARPES currently
places restrictive version requirements on a few libraries. If you find these too restrictive, 
[drop us a note](mailto:chstansbury+arpes@gmail.com) and we will see if we can help. Some common failure modes have 
been written up in the [FAQ](/faq).

After you've installed, you can run a few self checks and feature gates with

```python
import arpes
arpes.check()
```

### Requirements

PyARPES is [tested and installable](https://dev.azure.com/lanzara-group/PyARPES) on Windows 10, Ubuntu + similar 
linuxes, and Mac OS X. Currently Python 3.6 and 3.7 are verified compatible, but there should be no fundamental issues 
preventing use with Python 3.4+.

We currently require the following strict versions

```python
tornado==4.5.3     # for Bokeh
xarray==0.9.6      # due to Issue 2097 (https://github.com/pydata/xarray/issues/2097)
h5py==2.7.0        # avoids an OS X bug, but should be safe to relax on other systems
matplotlib>=3.0.3  # we require matplotlib 3
bokeh==0.12.10     # Irrelevant if you don't use the Bokeh interactive tools
netCDF4==1.3.0     # Avoids another platform specific data-loading bug
```

You should have no problems if you install into a new Anaconda environment.

### From Package Managers

You can install PyARPES from PyPI

```bash
pip install arpes
```

or from the Anaconda package repositories through the `arpes` channel

```bash
conda install -c arpes arpes
```

If you want to install with `pip`, you will need to install also the platform specific libraries for `h5py` and `netCDF4`.
You can find details on the netCDF library [here (under the Install section)](https://unidata.github.io/netcdf4-python/netCDF4/index.html)
and for the HDF library [here](http://docs.h5py.org/en/latest/build.html).

### Installation from Source

Using an installation from source is the best option if you want to frequently change 
the source of PyARPES as you work, or you would like to contribute to the development 
of PyARPES. You can use code available either from the main repository at 
[GitLab](https://gitlab.com/lanzara-group/python-arpes.git) or the 
[GitHub mirror](https://github.com/chstan/arpes).

1. Make an Anaconda environment or `virtualenv` for your installation of PyARPES
2. Clone the respository

```bash
git clone https://gitlab.com/lanzara-group/python-arpes
```

or 

```bash
git clone https://github.com/chstan/arpes
```

3. Install requirements that are hard to manage with pip: `conda install -y h5py==2.7.0 netCDF4==1.3.0`
4. Install PyARPES into your conda environment `pip install -e .`


### Additional Suggested Steps

1.  Clone or duplicate the folder structure in the repository
    `arpes-analysis-scaffold`, skipping the example folder and data if
    you like
2.  Install and configure standard tools like
    [Jupyter](https://jupyter.org/) or Jupyter Lab. Notes on installing
    and configuring Jupyter based installations can be found in
    `jupyter.md`
3.  Explore the documentation and example notebooks at [the
    documentation site](https://arpes.netlify.com/).

# Contributing and Documentation

See the section on the docs site about
[contributing](https://arpes.netlify.com/#/contributing)
for information on adding to PyARPES and rebuilding documentation from
source.

Copyright © 2018-2019 by Conrad Stansbury, all rights reserved.
