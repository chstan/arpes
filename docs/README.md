# PyARPES

**December 2020, V3 Release**: 
The current relase focuses on improving usage and workflow for less experienced Python users, lifting version incompatibilities with dependencies, and ironing out edges in the user experience. 

For the most part, existing users of PyARPES should have no issues upgrading, but we now require Python 3.8 instead of 3.7. We now provide a conda environment specification which makes this process simpler, see the installation notes below. It is recommended that you make a new environment when you upgrade.

<figure>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/Gd0qJuInzvE" frameborder="0" 
          allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
  </iframe>
  <figcaption>You can find more usage and example videos <a href="/#/example-videos">here</a>.</figcaption>
</figure>

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
files from Igor Pro (see in particular the section on [importing Igor Data](/igor-pro)), NeXuS files, and others. 
Additional data formats can be added via a user plugin system.

If PyARPES helps you in preparing a conference presentation or publication, please respect the guidelines 
for citation laid out in the notes on [user contribution](/contributing). Contributions and suggestions from the 
community are also welcomed warmly.

#### Tool Development

Secondary to providing a healthy and sane analysis environment, PyARPES is a testbed for new analysis and 
correction techniques, and as such ships with `scikit-learn` and `open-cv` as compatible dependencies for 
machine learning. `cvxpy` can also be included for convex optimization tools.


## Installation

Some common issues in installation have been written up in the [FAQ](/faq). 

You can install PyARPES in an editable configuration so that you can edit it to your needs (recommended) or as a standalone package from a package manager. In the latter case, you should put any custom code in a separate module which you import together with PyARPES to serve your particular analysis needs.

### Installation from Source

Using an installation from source is the best option if you want to frequently change 
the source of PyARPES as you work. You can use code available either from the main repository at 
[GitLab](https://gitlab.com/lanzara-group/python-arpes.git) or the 
[GitHub mirror](https://github.com/chstan/arpes).

1. **Install Miniconda or Anaconda** according to the [directions](https://docs.conda.io/en/latest/miniconda.html)
2. Clone or otherwise download the respository

```bash
git clone https://gitlab.com/lanzara-group/python-arpes
```
3. Make a conda environment according to our provided specification

```bash
cd path/to/python-arpes
conda env create -f environment.yml
```

3. Activate the environment

```bash
conda activate arpes
```

4. Install PyARPES in an editable configuration

```bash
pip install -e .
```

5. _Recommended:_ Configure IPython kernel according to the **Barebones Kernel Installation** below

### From Package Managers

It is highly recommended that you install PyARPES through `conda` rather than `pip`. You will also need to specify 
`conda-forge` as a channel in order to pick up a few dependencies. Make sure you don't add conda-forge with higher priority 
than the Anaconda channel, as this might cause issues with installing BLAS into your environment. We recommend

```bash
conda config --append channels conda-forge
conda install -c arpes arpes
```

### Additional Suggested Steps

1.  Install and configure standard tools like
    [Jupyter](https://jupyter.org/) or Jupyter Lab. Notes on installing
    and configuring Jupyter based installations can be found in
    `jupyter.md`
3.  Explore the documentation and example notebooks at [the
    documentation site](https://arpes.netlify.com/).
    
    
### Barebones kernel installation

If you already have Jupyter and just need to register your environment. You can do
```bash
pip install ipykernel
python -m ipykernel install --user 
```

You can also give the kernel a different display name in Juptyer with 
`python -m ipykernel install --user --display-name "My Name Here"`.

# Contributing and Documentation

See the section on the docs site about
[contributing](https://arpes.netlify.com/#/contributing)
for information on adding to PyARPES and rebuilding documentation from
source.

Copyright © 2018-2020 by Conrad Stansbury, all rights reserved.
Logo design, Michael Khachatrian
