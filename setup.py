#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Specifies installation requirements and build steps."""

import io
import os

from setuptools import find_packages, setup
from setuptools.command.install import install

NAME = "arpes"
DESCRIPTION = "Modular data analysis code for angle resolved photoemission spectroscopy (ARPES)"
URL = "https://gitlab.com/lanzara-group/python-arpes"
EMAIL = "chstan@berkeley.edu"
AUTHOR = "Conrad Stansbury"
REQUIRES_PYTHON = ">=3.8.0,<3.9"  # we're being less permissive because of pyqtgraph

about = {}
with open("./arpes/__init__.py") as fp:
    exec(fp.read(), about)

VERSION = about["VERSION"]

DEPENDENCY_GROUPS = {
    "core": [
        "astropy",
        "xarray>=0.16.1",
        "h5py>=3.2.1",
        "pyqtgraph>=0.12.0,<0.13.0",
        "PyQt5==5.15",
        "netCDF4>=1.5.0,<2.0.0",
        "colorcet",
        "pint",
        "pandas",
        "numpy>=1.20.0,<2.0.0",
        "scipy>=1.6.0,<2.0.0",
        "lmfit>=1.0.0,<2.0.0",
        "scikit-learn",
        # plotting
        "matplotlib>=3.0.3",
        "bokeh>=2.0.0,<3.0.0",
        "ipywidgets>=7.0.1,<8.0.0",
        # Misc deps
        "packaging",
        "colorama",
        "imageio",
        "titlecase",
        "tqdm",
        "rx",
        "dill",
        "ase>=3.20.0,<4.0.0",
        "numba>=0.53.0,<1.0.0",
    ],
    "igor": ["igor==0.3.1"],
    "ml": [
        "scikit-learn>=0.24.0,<1.0.0",
        "scikit-image",
        "cvxpy",
        "libgcc",
    ],
}

requirements = [y for k, v in DEPENDENCY_GROUPS.items() for y in v if k not in {"igor", "ml"}]

DEV_DEPENDENCIES = {
    "jupyter": [
        "jupyter",
        "ipython",
        "jupyter_contrib_nbextensions",
        "notebook>=5.7.0",
    ],
    "test": [
        "attrs==17.4.0",
        "pluggy==0.6.0",
        "py==1.5.2",
        "pytest==3.3.2",
        "setuptools==38.4.0",
    ],
}


with open("./pypi-readme.rst", "r") as f_readme:
    long_description = f_readme.read()


DOCUMENTATION_URL = "https://arpes.readthedocs.io/"

POST_INSTALL_MESSAGE = """
Documentation available at: {}

You should follow standard best practices for working with IPython and Jupyter.

To get the interactive volumetric data explorer `qt_tool` you will need to install
`PyQt5` and `pyqtgraph`. 

To use the Igor data loading libraries in PyARPES you will need to install the `igor` 
module from 'https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1'.

Some functionality, including PCA/Factor Analysis decomposition tools, require 
additional heavy dependencies such as `scikit-learn` and `scikit-image`. 

For Jupyter integration, please have a look at the documentation (link above).
For support issues, contact chstansbury@gmail.com or chstan@berkeley.edu.
""".format(
    DOCUMENTATION_URL
)

packages = find_packages(
    exclude=(
        "tests",
        "source",
        "info_session",
        "scripts",
        "docs",
        "example_configuration",
        "conda",
        "figures",
        "exp",
        "datasets",
        "resources",
    )
)

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = "\n" + f.read()


class PostInstallCommand(install):
    """Provides some extra information and context after install."""

    def run(self):
        """Print the post-installation message after successful install."""
        install.run(self)
        print(POST_INSTALL_MESSAGE)


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=packages,
    dependency_links=[
        "https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1",
    ],
    install_requires=requirements,
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 7",
        "Operating System :: Microsoft :: Windows :: Windows 8",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Unix",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # $ setup.py publish support.
    cmdclass={
        "install": PostInstallCommand,
    },
)
