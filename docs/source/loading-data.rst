.. _loading-data:

Loading Data
============

The only data loading interface you need to be familiar with is ``arpes.io.load_data``.
This function has several usage patterns for loading data by associated workspace, 
by "file number", or by absolute or relative paths.

Additionally, you can opt to specify the ARPES setup used to collect the data, or it can be 
inferred in certain circumstances.

My files are numbered
~~~~~~~~~~~~~~~~~~~~~

You can just use the numeric fragment identifying your data. For
instance if you wanted to load the data ``data/my_data_005.pxt``, you
can use

.. code:: python

   from arpes.io import load_data

   load_data(5, location="BL7")  # loads as MAESTRO microARPES

My files are numbered, and I want to leave the location unspecified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can leave the `location=` kwarg unspecified in the call to `load_data`.
It's best to provide the `location` kwarg if you know it, because otherwise PyARPES
will have to sequentially try loaders (filtered by filetype) to find one that works.

.. code:: python

   from arpes.io import load_data

   load_data(5)  # we don't pass `location=` here, PyARPES will try to figure it out

My files are not numbered
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the path to the data, either as a string or a `Path` instance.

.. code:: python

   from arpes.io import load_data
   from pathlib import Path

   # these are equivalent
   load_data('/path/to/my/data.fits', location='ALS-BL7')
   load_data(Path('/path/to/my/data.fits'), location='ALS-BL7')

I want to use a specific plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you define your own plugin but you haven't registered it, you can provide it
via the `location` kwarg. For example, we will manually load MAESTRO nano-ARPES data

.. code:: python

   from arpes.io import load_data
   from arpes.endstations.plugins.MAESTRO import MAESTRONanoARPESEndstation

   # equivalent in this case to passing `location="BL7-nano"`
   load_data("path/to/my/data.fits", location=MAESTRONanoARPESEndstation)


This is helpful if you defined a loading plugin in local module and want to use it.

I want to perform post-loading steps for every piece of data in an analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, just write a wrapper loading function for a given workspace or project
and use it to apply some additional steps to process your data.

Let's imagine we just want to add an attribute with the date of the analysis.

.. code:: python

   from datetime import datetime
   from arpes.io import load_data

   def load_with_date(file, location: str = None):
      """Attach the current datetime when loading."""
      data = load_data(file, location)
      data.attrs["analysis_date"] = datetime.now().isoformat()

      return data

   # now, we can use it just like `load_data`
   load_with_date("path/to/my_data.h5", location="BL7-nano")

This example is artificial, but you can use this pattern to apply corrections
or perform other cumbersome steps.

I’m still running into an issue, or my use case doesn’t fit nicely
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take a look at the `frequently asked questions </faq>`__ or get in
contact on the `GitLab Issues
Page <https://gitlab.com/lanzara-group/python-arpes/issues>`__ and we
will be happy to help.
