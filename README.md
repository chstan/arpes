# Installation

This library requires a copy of Python 3.5 in order  to function, as well as installation of the dependencies listed 
in the ```requirements.txt``` file. Steps are outlined below for installation as well as optional steps to make the
user experience better.

#### Required Steps

1. Python with managed environments
2. Installation of this package
3. Project dependencies
4. Creation of a ```local_config.py``` file

#### Optional Steps

1. Jupyter
2. IPython Kernel Customization
3. Setting up data, analysis notebooks, and datasets


## Installing Python through Conda

1. Install `conda` through the directions listed at [Anaconda](https://www.anaconda.com/download/), you want the Python 
3 version.
2. Create a Python 3.5 environment using the command ``conda create -n {name_of_your_env} python=3.5 scipy jupyter``. 
You can find more details in the conda docs at the 
[Conda User Guide](https://conda.io/docs/user-guide/tasks/manage-environments.html).
3. In order to use your new environment, you can use the scripts `source activate {name_of_your_env}` 
and `source deactivate`. Do this in any session where you will run your analysis or Jupyter.  

## Installation of this Package

Clone this repository somewhere convenient.

## Project Dependencies

Inside the cloned repository, after activating your environment, run `pip install -r requirements.txt`. This will
install all packages required by the code. If you ever want to add a new package you find elsewhere, just 
`pip install {package_name}` and add it to `requirements.txt`. 

## `local_config.py`

You need to tell the analysis code where your data and datasets are. To do this you need to make a 
Python file called `local_config.py` and put it into the `arpes` folder. Inside this file you 
should define the two variables `DATA_PATH` and `DATASET_CACHE_PATH`. These tell the project
where to look for data, and where to put cached computations, respectively. For reference, 
Conrad's looks like:

```
DATA_PATH = '/Users/chstansbury/Research/lanzara/data/'
DATASET_CACHE_PATH = '/Users/chstansbury/Research/lanzara/data/cache/'
```


## Jupyter

You should already have Jupyter if you created an environment with `conda` according to the above. Ask Conrad
about good initial settings.

## IPython Kernel Customization

Ask Conrad about getting your kernel set up so that you don't have to import common code into 
notebooks. 

# Getting Started with Analysis

Ask Conrad! He will put some tutorial/example notebooks he has collected eventually.