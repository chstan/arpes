# Installation

This library requires a copy of Python 3.5 in order  to function, as well as installation of the dependencies listed
in the ```requirements.txt``` file. Steps are outlined below for installation as well as optional steps to make the
user experience better.

#### Required Steps

0. (Shell environment for OS X/Unix, windows users see section below)
1. Python with managed environments
2. Installation of this package
3. Project dependencies
4. Creation of a ```local_config.py``` file
5. Installation of Jupyter
6. Configuration of the IPython Kernel

#### Optional Steps

1. Setting up data, analysis notebooks, and datasets
2. If you are using Jupyter Lab as opposed to Jupyter, installation
   of the nbextensions: to do this install nodejs from nodejs.org and restart
   Then open a terminal and run the labextension installation

The rest of this document will guide you through the process of getting through the first set of steps. Additionally, example configuration files can be found in `example_configuration`.

## Shell environment for Windows Users (optional)

If you are running Windows 10 you can do the following

1. Open PowerShell as administrator (right click menu)
2. Run `Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux`
3. Restart computer when prompted

If you are not running Windows 10, install [MSYS2](https://www.msys2.org) or [Cygwin](https://www.cygwin.com/). MSYS2 is recommended.

## Installing Python through Conda

1. Install `conda` through the directions listed at [Anaconda](https://www.anaconda.com/download/), you want the Python
3 version.
2. On UNIX like environments, create a Python 3.5 environment using the command ``conda create -n {name_of_your_env} python=3.5 scipy jupyter``.
You can find more details in the conda docs at the
[Conda User Guide](https://conda.io/docs/user-guide/tasks/manage-environments.html). For Windows Users, launch Anaconda Navigator and create an environment.
3. In order to use your new environment, you can use the scripts `source activate {name_of_your_env}`
and `source deactivate`. Do this in any session where you will run your analysis or Jupyter. For Windows Users or users of
graphical conda, launch your environment from the Navigator.

## Installation of this Package

Clone this repository somewhere convenient. Remember the path (i.e. full directory from the root of your hard disk) where you installed it.

## Project Dependencies

Inside the cloned repository, after activating your environment, install dependencies:

1. Install `xrft` via `conda install -c conda-forge xrft`
2. Run `pip install -r requirements.txt`. This will
install all packages required by the code. If you ever want to add a new package you find elsewhere, just
`pip install {package_name}` and add it to `requirements.txt`. Conda Navigator users can get a terminal by launching Jupyter
and requesting `> New > Terminal Session` from the Jupyter [chrome](https://en.wikipedia.org/wiki/Graphical_user_interface#User_interface_and_interaction_design).


## `local_config.py`

The local configuration allows you to override the settings that are committed to the repository and therefore shared.
You can use this to change (somewhat) where data is stored, as well as adjust settings on various interactive tools.
For reference, Conrad's looks like

```
DATA_PATH = None
SETTINGS = {
    'interactive': {
        'main_width': 600,
        'marginal_width': 300,
        'palette': 'magma',
    },
}
```

## Environment

Please read to the bottom of this section for an alternative. You need to make sure to export a variable ``ARPES_ROOT`` in order to run scripts. If you are on a UNIX-like system
you can add the following to your ``.bashrc``, ``.bash_profile`` or equivalent:

```bash
export ARPES_ROOT="/path/to/wherever/you/installed/this/project/"
```

The value of ``ARPES_ROOT`` should be defined so that ``$ARPES_ROOT/README.md`` points to the file that you
are currently reading.

To make using the code simpler, consider an alias to move to the data analysis location and to start the
virtual environment. On Conrad's computer this looks like:

```bash
alias arpes="cd $ARPES_ROOT && source activate python_arpes"
alias arpesn="cd $ARPES_ROOT && source activate python_arpes && jupyter notebook"
```

Similar commands should be placed in your `.bashrc`.

Alternatively, you can effectively set an environment variable by creating a file in your `IPython` startup folders
that sets it through `os.environ`. On Windows this might look like `00-set-env.py`:

```
import os
os.environ['ARPES_ROOT'] = r'C:\some\path\to\installation\of\python-arpes'
```

## Jupyter

You should already have Jupyter if you created an environment with `conda` according to the above. Ask Conrad
about good initial settings.

## IPython Kernel Customization

If you don't want to have to import everything all the time, you should customize your IPython session so that it
runs imports when you first spin up a kernel. There are good directions for how to do this online, but a short
version is:

1. Create an IPython profile, use this to start your notebooks
2. In ``~/.ipython/profile_default/`` make a folder `startup`
3. Add the files ``~/.ipython/profile_default/startup/00-add-arpes-path.py`` and
``~/.ipython/{Your profile}/startup/01-common-imports.ipy`` according to the templates in `ipython_templates`. See in particular
note above about setting the environment variable using this file.
4. Customize to your liking

Note that you can customize the default profile or a different if you wish instead.

It is important that the filenames you put are such that ``-add-arpes-path`` is lexographically first, as this ensures
that it is executed first. The ``.ipy`` extension on ``01-common-imports.ipy`` is also essential.
Ask Conrad if any of this is confusing.

# Getting Started with Analysis

Ask Conrad! Also look in `datasets/example`. Also feel free to contribute examples.
