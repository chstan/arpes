# Get Started with PyARPES

## Checking your installation

Some features in PyARPES require libraries that are not installed by default, 
either because they are heavy dependencies we don't want to force on users, or there 
are possible issues of platform compatibility.

You can check whether your installation in a Python session or in Jupyter

```python
import arpes
arpes.check()
```

You should see something like this depending on the state of your optional dependencies:

```text
[✘] Igor Pro Support:
	For Igor support, install igorpy with: 
	pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1
[✔] Bokeh Support
[✔] qt_tool Support
[✔] Import almost everything in PyARPES
```

## Loading example data

At this point, you should be able to load the example data, an ARPES spectrum of 
the topological insulator bismuth selenide:

```python
from arpes.io import load_example_data
load_example_data()
```

## Loading your own data

If you have the path to a piece of data you want to load as well as the data source it 
comes from (see the section on [plugins](/writing-plugins) for more detail), you can load it
with `arpes.io.load_without_dataset`:

```python
from arpes.io import load_without_dataset
load_without_dataset('/path/to/my/data.h5', location='ALS-BL7')
```   

**Note**: Although you can do all your analysis this way by loading data directly and 
specifying  the originating experiment, PyARPES works better and can keep figures and 
data organized if you opt into using [workspaces](/workspaces).  


## A minimal and useful set of imports

Although you can manage and directly import everything from PyARPES as you need it, 
you can also import all the useful bits into your global scope up front, which makes 
working in Jupyter and similar environments a bit more convenient.

```python
import arpes
arpes.setup(globals(), '/path/to/your/datasets/')
```

The second argument is the location of your workspaces tree, if you use this approach, you 
can read more about this in the last section of the [workspaces documentation](/workspaces).  

## What's next?

With the example data in hand, you can jump into the rest of the examples on the site. 
A good place to start is on the section for [exploration](/basic-data-exploration) of 
ARPES data. 


