# Loading Data Ad-Hoc

There are several ways to load data that circumvent workspaces.
The most straightforward is to call `arpes.io.load_without_dataset` manually,
or to call `arpes.io.fallback_load`/`arpes.io.fld` manually.

Some patterns for invocation are discussed below.

### My files are numbered and the data source is unambiguous

You can just use the numeric fragment identifying your data.
For instance if you wanted to load the data
`data/my_data_005.pxt`, you can use

```python
from arpes.io import load_without_dataset

load_without_dataset(5)
```

or 

```python
from arpes.io import fld
fld(5)
```

### My files are numbered, but the source is not disambiguated

Sometimes several data loaders might be able to load
a particular piece of data. Examples include structured text,
or if you subclass a more generic plugin for your lab's DAQ software.

In this case you need to pass the `location` keyword.

```python
from arpes.io import load_without_dataset

load_without_dataset(5, location='My PXT Plugin') # <- fake plugin
```

and similarly for `fld`.

### My files aren't numbered

Pass the full path and the `location` keyword to
`load_without_dataset`. For instance:

```python
from arpes.io import load_without_dataset
load_without_dataset('/path/to/my/data.h5', location='ALS-BL7')
```  

### I'm still running into an issue, or my use case doesn't fit nicely

Take a look at the [frequently asked questions](/faq) or get in contact on the
[GitLab Issues Page](https://gitlab.com/lanzara-group/python-arpes/issues) and we will
be happy to help.