# Setting up metadata for an expeiment from an Excel file

## Need 

The analysis procedures contained in this project
rely on the existence of a structured set of
file metadata in order to know how to automate common
analysis procedures. 

ARPES metadata gets recorded in many formats:

1. Excel files recorded during the experiment by the experimenter
2. Metadata attached to scans in FITS/SESb headers, recorded by the instrumentation
3. Metadata separate from scans but semantically attached, recorded by the instrumentation

These scripts "canonicalize" the metadata by making it available 
in two places:

1. In a structured Excel file, with one row per experimental run
2. In file attached metadata stored with the scan, with scans
being of a universal file format ``HDF5``. 

In practice, the fact that this information is stored in two places is immaterial.
When loaded data, the two data sources are verified against one another to produce
a "single source of truth", and all information is eventually available on the
``dataset.S.{attr}`` accessors.

## Advice

You should add these scripts to your ``PATH`` so that they are more readily
accessible. On UNIX like systems this can be achieved by adding the following line 
to your ``.bashrc`` or equivalent:

```bash
export PATH="$PATH:$HOME/path/to/analysis/root/scripts/"
``` 

The fragment ``/path/to/analysis/root/`` should be replaced by
the corresponding directory where you installed this analysis framework.

## Canonicalization

Two steps are performed to canonicalize data. The first is that your experiment logs (``.excel`` files)
are transformed into a cleaned dataset file that the analysis scripts can read. This file is given 
the extension ``.cleaned.excel``. The script that does this is ``clean_dataset.py``.
You can also use the function ``datasets.py:clean_xlsx_dataset`` which is used internally.
It is important that this step is performed first, because one of the effects is to produce
unique, random IDs for the datasets that are used to track the relationships between datasets
as part of the analysis process. For instance, these IDs are used for dependency tracking when
cacheing expensive computations like k-space conversions, as well as producing data provenance
ffor plots and analysis histories. The IDs are also used in the composition of analysis "pipelines".

The second step is to load the data into a universal format. The script that performs this process is 
``load_all_files.py``. This script loops through the datasets in the cleaned dataset description
and attempts to load the data from whatever format it was recorded in into HDF5. It also
attaches inferred scan information like the "scan type".

After both these steps have been performed, you can use the function ``default_dataset()`` 
in order to fetch a cleaned and parsed copy of the experimental metadata. This dataset can be filtered,
queried, or loaded from in order to perform analysis on relevant parts of your data.

### Using ``clean_dataset``

You can optionally skip this step and drop the `-uc` flag from the 
first invokation of ``load_all_files``. See below for example usage.

Using ``clean_dataset -w WORKSPACE`` should be enough. You might have to specify
how many rows to skip in order to reach the header of the Excel document if you 
use a complicated template for your experiment logs. You can look at the help info
for this command at ``clean_dataset --help`` for full information.

After you run this, have a look at the cleaned Excel document just to verify
everything looks alright. It should be pretty obvious if there is an error,
since data will be missing or the columns will be incorrect.  

### Using ``load_all_files``

Because Conrad is lazy you actually have to run ``load_all_files``
twice. It doesn't actually load the data twice, rather it is perhaps poorly 
named. The first time it is run it loads the data, and the second time it 
goes through the data to attach information to the dataset and to reconcile
information between the dataset `.excel` file and the scans themselves.

You can use the flag `-w WORKSPACE` to specify the workspace, or you can set your
global default workspace if you are only working on one project at a time in your
``local_config.py``. 

### First run to load data

Use the flag `-uc -l`.

### Second run to attach columns and reconcile metadata

Use the flags `-uc -c`. The first flag indicates to use the cleaned copy
of the dataset ``.excel``, while the second indicates reconciling data.

### Example Usage:

For a dataset ``datasets/RhSn2/``

```bash
clean_dataset -w RhSn2 --any-other-flags-necessary-here
load_all_files -w RhSn2 -uc -l
load_all_files -w RhSn2 -uc -c
``` 

You can also use the full flag names, described in the 
help for the command ``load_all_files --help``.

If you didn't use ``clean_dataset``, you can use

```bash
load_all_files -w RhSn2 -l
load_all_files -w RhSn2 --uc -c
```

## Producing a single source of truth

When running analysis, you want to be able to access all the information
about a particular scan, and you don't want to have to think about where to get it.

If you use ``scan = load_dataset(dataset_id, [optional dataset])``, this process is performed
for you automatically. All the information is available at `scan.S.[attrs]`.

# Keeping Data in Sync

These scripts also contain a utility ``catchup.py`` which is used to sync data
between your folders and the folders on the Lanzara drive.

## Using ``catchup.py``

``catchup.py`` uses a file ``drive.refs`` which should contain a string that specifies
where to sync data from. As an example, in my dataset folder for MoWTe2 work, I have
a ``drive.refs`` file with the contents

```text
ALG_Chamber/2017/20171206_MoWTe
```

This specifies to look in the folder ``ALG_Chamber/2017/20171206_MoWTe``, and to 
put datasets in the datasets folder for MoWTe2, and data in the data folder
for the project. If you put multiple lines in ``drive.refs`` all the locations
will be synced to your project folder for the specified project.

The ``drive.refs`` folder should be placed in the dataset folder, i.e.the one
containing the excel files for the project. You should also run ``catchup.py``
with the ``-w`` workspace flag. This indicates which project is associated.
A full example then is ``catchup -w MoWTe2`` inside the folder with the ``drive.refs``
file.