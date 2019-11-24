# Frequently Asked Questions

## Igor Installation

### Using the suggested invokation I get a pip error

Pip on Windows appears not to like certain archival formats.
While

```pip
 pip install https://github.com/chstan/igorpy.git#egg=igor-0.3.1
```

should work on most systems, you can also clone the repository:

```git
git clone https://github.com/chstan/igorpy
```

And then install into your environment from inside that folder.

```pip
(my arpes env) > echo "From inside igorpy folder"
(my arpes env) > pip install -e .
```

## Common Issues

### I want to use the Bokeh based interactive tools, what version of Tornado should I use?

Typically users run into an incompatibility around tornado (manifesting as a massive
red stacktrace when an attempt is made to call `.S.show`) when they are running
Jupyter from the same environment they installed PyARPES into.

You should install Jupyter into its own environment (I use a separate `[jupyter]` conda environment)
and register your analysis environment as a named kernel with Jupyter.

Anaconda is amazing but can be frustrating: I've found through painful experience
that it is best to avoid all changes to the root environment and to use Jupyter through a separate
dedicated environment. This also helps break dependencies between environments and makes
experimenting with another Jupyter version simple: you just have to make a new trial environment
before you switch.

### I tried to upgrade a package and now things aren't working... how do I get my old code working again?

For large upgrades I recommend making a new environment until you are sure you don't encounter issues (500 MB disk is cheap!).

It is also helpful to keep a record of "working" configurations on systems that you use. Different package managers have better
and worse ways of dealing with this, but you can typically recover a full installation of complex Python software with a list of the requirements
and their versions and the version for the interpreter. As a result,
I make a point to save a copy of my full requirements

```bash
$ pip freeze > working-dependencies-py38-date-14-11-2019.txt
```

You can then pip or conda install from this requirements file.

If you don't find this satisfying, you are probably a reasonable and sane human being. Packaging software is apparently more difficult than it 
would ideally be but the situation is improving.
