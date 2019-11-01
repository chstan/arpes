# Loading and Normalizing Data

## Loading data directly

In the previous section, we discussed how you can access data directly if you know the path to the file with
`arpes.io.load_without_dataset`. This is the most barebones way to read data into PyARPES.

```python
from arpes.io import load_without_dataset
load_without_dataset('/path/to/my/data.h5', location='ALS-BL7')
```  

PyARPES will also try to do its best if you don't provide a spreadsheet or location. For instance, 
if your data is in the folder `data` with the name `my_scan_1.fits`, then calling

```python
from arpes.io import load_without_dataset
load_without_dataset(1)
```

or 

```python
from arpes.io import fld
fld(1)
```

will load your data according to the most approriate supported `.fits` plugin: in this case the MAESTRO plugin.

## Loading a single file

If you use [workspaces](/workspaces), data can be loaded using its index in a spreadsheet or canonical (normalized) UUID. This programmatic interface 
allows for expressive analysis scripts that can perform a particular data analysis across all of your cuts, 
or across an experimental degree of freedom like the sample temperature.

Optionally, you can pass `fld` or its siblings a `workspace=` argument to load data
from a different workspace.

Unless you normalize your data, you should prefer the function `fld` which will work 
transparently even from the original data files.  

![Loading a file](static/ld.png)

As we can see, the loaded data takes the form of an xarray `Dataset`. For those coming from a Python background,
xarray objects are very much like pandas' `DataFrames`, except that they can be multidimensional. For those from
an Igor background, `Dataset`s and `DataArray`s are very similar to waves, except that they can host attached 
attributes, allow for more expressive selection and manipulation (see the [following section](/basic-data-exploration)),
and have axes labelled not by a number but by a physically significant name (here, the hemisphere axes: 'eV', 'phi').

Read more about the [data format for spectra in PyARPES](/spectra) or continue to the section on 
[basic data exploration](/basic-data-exploration).

## Loading Several Files

You can also use `arpes.io.stitch` in order to combine several pieces of data along an axis at load time.