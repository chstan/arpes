#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.command.install import install

NAME = 'pypes'
DESCRIPTION = 'Modular data analysis code for angle resolved photoemission spectroscopy (ARPES)'
URL = 'https://gitlab.com/lanzara-group/python-arpes'
EMAIL = 'chstan@berkeley.edu'
AUTHOR = 'Conrad Stansbury'
REQUIRES_PYTHON = '>=3.5.0'

about = {}
with open('./arpes/__init__.py') as fp:
    exec(fp.read(), about)

VERSION = about['VERSION']

with open('requirements.txt', 'r') as f_requirements:
    requirements = f_requirements.read()

with open('README.rst', 'r') as f_readme:
    long_description = f_readme.read()


DOCUMENTATION_URL = "https://stupefied-bhabha-ce8a9f.netlify.com/#/"

POST_INSTALL_MESSAGE = """
Documentation available at: {}
""".format(DOCUMENTATION_URL)

packages = find_packages(exclude=('tests', 'source', 'info_session', 'docs', 'example_configuration',
                                  'figures', 'exp', 'datasets', 'resources',))

REQUIRED = [
    'tornado==4.5.3',
    # Data loading
    'astropy',
    'igor==0.3.1', # patched on GitHub
    'xarray==0.9.6',
    'h5py==2.7.0',
    'netCDF4==1.3.0', # some dependency bugs here and in h5py, so fix the versions
    'colorcet',

    # Analysis
    'pint',
    'pandas',
    'dask',
    'numpy',
    'scipy',
    'lmfit',
    'scikit-learn',
    'scikit-image',
    'xrft==0.1.dev',

    # Plotting
    'matplotlib',
    'seaborn',
    'bokeh==0.12.10', # will look into upgrading to >=1.0.3 at a later date
    'PyQt5',
    'pyqtgraph',

    # Rendering and exploration
    #'ipywidgets==7.0.1',

    # Dependencies
    'xlrd',
    'titlecase',
    'openpyxl',
    'toolz>=0.7.3', # required by dask and delayed, inconveniently, since only 'distributed' does not include it

    # UI
    'tqdm',
]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()


class LinkIPythonCommand(Command):
    """
    Links user default iPython to template files
    Largely defunct and superceded by just using setuptools to install the package
    and making a single "import useful stuff" function available for easy of use.
    """

    description = 'Establish symbolic links between template iPython file, establish path.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        sys.exit()
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['VERSION']))
        os.system('git push --tags')

        sys.exit()


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

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    dependency_links=[
        'https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1',
        # use this specific 'xrft' since it still has `_hanning` and we have not updated to use the new `_create_window`
        'https://github.com/xgcm/xrft/tarball/879643cb0d6779632fc7600876bd90200a632028#egg=xrft-0.1.dev',
    ],
    install_requires=REQUIRED,

    include_package_data=False, # until we get a manifest file

    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: Closed',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
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
