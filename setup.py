#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import re
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.command.install import install

NAME = 'arpes'
DESCRIPTION = 'Modular data analysis code for angle resolved photoemission spectroscopy (ARPES)'
URL = 'https://gitlab.com/lanzara-group/python-arpes'
EMAIL = 'chstan@berkeley.edu'
AUTHOR = 'Conrad Stansbury'
REQUIRES_PYTHON = '>=3.5.0'

about = {}
with open('./arpes/__init__.py') as fp:
    exec(fp.read(), about)

VERSION = about['VERSION']

DEPENDENCY_GROUPS = {
    'core': [
        'tornado==4.5.3', # this version required due to Bokeh weirdness
        'astropy',

        # this version required as 0.10.1 introduces a change that forced all assignment
        # through Variable.get_compatible_data, which coerces to an array
        'xarray==0.9.6',

        'h5py==2.7.0',
        'netCDF4==1.3.0',
        'colorcet',

        'pint',
        'pandas',
        'dask',
        'numpy',
        'scipy',
        'lmfit>=0.9.13',
        'scikit-learn',

        # plotting
        'matplotlib>=3.0.3',
        'seaborn',
        'bokeh==0.12.10',
        'ipywidgets==7.0.1',

        # Misc deps
        'xlrd',
        'colorama',
        'titlecase',
        'openpyxl',
        'tqdm',
    ],
    'igor': ['igor==0.3.1'],
    'ml': [
        'scikit-learn',
        'scikit-image',
        'cvxpy',
        'libgcc',
    ],
}

requirements = [y for k, v in DEPENDENCY_GROUPS.items() for y in v if k not in {'igor', 'ml'}]

DEV_DEPENDENCIES = {
    'jupyter': [
        'jupyter',
        'ipython',
        'jupyter_contrib_nbextensions',
        'notebook>=5.7.0',
    ],
    'test': [
        'attrs==17.4.0',
        'pluggy==0.6.0',
        'py==1.5.2',
        'pytest==3.3.2',
        'setuptools==38.4.0',
    ]
}


with open('README.rst', 'r') as f_readme:
    long_description = f_readme.read()


DOCUMENTATION_URL = "https://stupefied-bhabha-ce8a9f.netlify.com/#/"

POST_INSTALL_MESSAGE = """
Documentation available at: {}

You should follow standard best practices for working with IPython and Jupyter.

To get the interactive volumetric data explorer `qt_tool` you will need to install
`PyQt5` and `pyqtgraph`. 

To use the Igor data loading libraries in PyARPES you will need to install the `igor` 
module from 'https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1'.

Some functionality, including PCA/Factor Analysis decomposition tools, require 
additional heavy dependencies such as `scikit-learn` and `scikit-image`. 
""".format(DOCUMENTATION_URL)

packages = find_packages(exclude=('tests', 'source', 'info_session', 'docs', 'example_configuration',
                                  'conda', 'figures', 'exp', 'datasets', 'resources',))

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()


class PostInstallCommand(install):
    """
    Provides some extra information and context after install.
    """

    def run(self):
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
        'https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1',
    ],
    install_requires=requirements,

    include_package_data=True,

    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: Microsoft :: Windows :: Windows 8',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # $ setup.py publish support.
    cmdclass={
        'install': PostInstallCommand,
    },
)
