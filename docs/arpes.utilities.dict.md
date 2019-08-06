# arpes.utilities.dict module

**arpes.utilities.dict.rename\_keys(d, keys\_dict)**

**arpes.utilities.dict.clean\_keys(d)**

**arpes.utilities.dict.rename\_dataarray\_attrs(arr:
xarray.core.dataarray.DataArray, \*args,**kwargs)\*\*

**arpes.utilities.dict.rename\_datavar\_attrs(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
\*args,**kwargs)\*\*

**arpes.utilities.dict.clean\_datavar\_attribute\_names(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
\*args,**kwargs)\*\*

**arpes.utilities.dict.clean\_attribute\_names(arr:
xarray.core.dataarray.DataArray, \*args,**kwargs)\*\*

**arpes.utilities.dict.case\_insensitive\_get(d: dict, key: str,
default=None, take\_first=False)**

> Looks up a key in a dictionary ignoring case. We use this sometimes to
> be nicer to users who don’t provide perfectly sanitized data :param d:
> :param key: :param default: :param take\_first: :return:
