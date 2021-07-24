Get Started with PyARPES
========================

Checking your installation
--------------------------

Some features in PyARPES require libraries that are not installed by
default, either because they are heavy dependencies we don’t want to
force on users, or there are possible issues of platform compatibility.

You can check whether your installation in a Python session or in
Jupyter

.. code:: python

   import arpes
   arpes.check()

You should see something like this depending on the state of your
optional dependencies:

.. code:: text

   [✘] Igor Pro Support:
       For Igor support, install igorpy with: 
       pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1
   [✔] Bokeh Support
   [✔] qt_tool Support
   [✔] Import almost everything in PyARPES

For the lazy, importing everything
----------------------------------

For convenience, we provide ``.all`` submodules that let you import what
is typically used in analysis. You can import more restrictive subsets
if you prefer.

.. code:: python

   # Will take a few seconds on kernel startup, but very convenient
   from arpes.all import *

Loading example data
--------------------

At this point, you should be able to load the example data, an ARPES
spectrum of the topological insulator bismuth selenide:

.. code:: python

   from arpes.io import load_example_data
   load_example_data()

Loading your own data
---------------------

If you have the path to a piece of data you want to load as well as the
data source it comes from (see the section on
`plugins </writing-plugins>`__ for more detail), you can load it with
``arpes.io.load_without_dataset``:

.. code:: python

   from arpes.io import load_data
   load_data('epath/to/my/data.h5', location='ALS-BL7')

What’s next?
------------

With the example data in hand, you can jump into the rest of the
examples on the site. If you’re a visual learner or are new to Jupyter
and are running into issues, have a look at the `tutorial
videos </example-videos>`__. Another good place to start is on the
section for `exploration </basic-data-exploration>`__ of ARPES data.
