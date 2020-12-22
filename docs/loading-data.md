# Loading and Normalizing Data

## Loading data directly

To load data in PyARPES, you should use `arpes.io.load_without_dataset`.

```python
from arpes.io import load_without_dataset
load_without_dataset('/path/to/my/data.h5', location='ALS-BL7')
```  

PyARPES will also try to do its best if you don't provide a full path to the data. For instance, 
if your data is in the folder `data` with the name `my_scan_1.fits`, then calling

```python
from arpes.io import load_without_dataset
load_without_dataset(1)
```

will load your data according to the most approriate supported `.fits` plugin: in this case the MAESTRO plugin.

## Loading Several Files

There's no magic here. You can load files programmatically, and then combine them as necessary.

```python
files = [load_without_dataset(i, location="MAESTRO") 
         for i in range(5)]

# do some concatenation with the results in `files`
```

Have a look at the `xarray` documentation for best practices on concatenating data.

You can also use `arpes.io.stitch` in order to combine several pieces of data along an axis at load time.