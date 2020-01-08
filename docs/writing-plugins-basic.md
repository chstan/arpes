# Adding Data Sources: Basic Plugins

This guided section will show you how to add a rudimentary data loading plugin as a demonstration of how
to extend PyARPES to allow you to work with your lab's data.

This is the first of a two-part section on data loading and plugins in PyARPES. If your needs are more advanced, 
you can see the [second page](/writing-plugins) for more details.

## Loading CSV Files into the PyARPES Format

For all analysis work, PyARPES assumes that the data to be manipulated is an 
[xarray datatype](http://xarray.pydata.org/en/stable/), typically an 
[xarrary.Dataset](http://xarray.pydata.org/en/stable/data-structures.html#dataset) or an 
[xarray.DataArray](http://xarray.pydata.org/en/stable/data-structures.html#dataarray).

Additionally, ARPES data must be labeled with enough [standard coordinates](/spectra) that we can
convert to momentum. Let's assume that our data comes formatted in two files, one providing the data `{name}.csv`, and
one the coordinates `{name}.coords.csv`. A standard data file might look like

```csv
Analyzer Spectrum
35,   41,   43,  112,  229,  433,  654,  584,  262,  105
89,  153,  207,  281,  529,  969, 1061,  602,  236,  101
295,  180,  249,  522,  833,  911,  856,  536,  236,   98
261,  226,  379,  509,  613,  787,  777,  522,  224,   94
271,  268,  338,  397,  568,  746,  703,  478,  217,   93
233,  204,  327,  464,  557,  691,  682,  477,  216,   93
185,  142,  203,  412,  681,  792,  732,  494,  223,   94
189,  130,  141,  206,  395,  740,  934,  615,  233,   96
146,  142,  151,  169,  238,  364,  531,  501,  267,  110
36,   40,   49,   89,  144,  214,  288,  256,  161,   96
```

and a standard coordinates file might hypothetically look like

```csv
energy angle
-0.425 0.221
-0.369 0.263
-0.313 0.305
-0.258 0.347
-0.202 0.389
-0.146 0.431
-0.090 0.472
-0.034 0.514
 0.020 0.556
 0.076 0.598
```

## Barebones, Function-based Approach

The simplest way to handle this task is just to write a data loading function that we can use to load the CSVs.
As a first pass, we can load just the data file and turn it into an ``xarray.DataArray``.

```python
import xarray as xr
import numpy as np

def load_csv_datatype(path_to_file: str) -> xr.DataArray:
    loaded_data = np.loadtxt(path_to_file, delimiter=',', skiprows=1) # skip the Data comment
    return xr.DataArray(loaded_data)
``` 

All we need to do now is attach the coordinates. Let's modify the function to load also the columns from the 
other file

```python
import xarray as xr
import numpy as np
from pathlib import Path

def load_csv_datatype(path_to_file: str) -> xr.DataArray:
    loaded_data = np.loadtxt(path_to_file, delimiter=',', skiprows=1) # skip the Data comment
    coordinates_file = str(Path(Path(path_to_file).stem + '.coords.csv').absolute())

    # get the dimension names
    with open(coordinates_file) as f:
        dim_names = f.readline().split()
    
    raw_coordinates = np.loadtxt(coordinates_file, skiprows=1)

    return xr.DataArray(
        loaded_data, 
        coords={d: raw_coordinates[:,i] for i, d in enumerate(dim_names)}, 
        dims=dim_names,
        # attrs={...} <- attributes here
    )
```

## Writing the Plugin

You can use the above code for loading this data, with the caveat mentioned above about momentum conversion.
Alternatively, we can integrate it into a plugin, which allows registering the data loading code against
a labeled "location" sourcing the data, and makes it easier to fill in missing values, ensure a standard representation,
and modify behavior between similar but differing data formats.

To do this, we subclass ``arpes.endstations.SingleFileEndstation``


```python
...
from arpes.endstations import SingleFileEndstation, add_endstation

class CSVDataEndstation(SingleFileEndstation):
    PRINCIPAL_NAME = 'csv' # allows us to use this code to refer to data labeled with location="csv"

    _TOLERATED_EXTENSIONS = {'.csv',} # allow only .csv files!
    
    def load_single_frame(self, frame_path: str=None, scan_desc: dict = None, **kwargs):
        data = load_csv_datatype(frame_path)
        return xr.Dataset({'spectrum': data})


# register it
add_endstation(CSVDataEndstation)
```

Now, you can load code with ``CSVDataEndstation.load_from_path``, or with ``CSVDataEndstation.load``. Additionally,
you can load using the standard data loading function by passing ``location='csv'``. Because ours is the only one
registered against the .csv file format, loading data without the location keyword will use our new class by default. 

The data loading plugins provide a number of features making it simpler to write data loading code for ARPES,
especially in normalizing coordinate units (mm for all distances, rad for all angular measures), and ensuring
the coordinates necessary to allow momentum conversion are attached. If you want to learn more about writing data
plugins, have a look at the in depth description of how they work in the [second part](/writing-plugins) of this
tutorial.  