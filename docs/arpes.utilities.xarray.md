# arpes.utilities.xarray module

**arpes.utilities.xarray.apply\_dataarray(arr:
xarray.core.dataarray.DataArray, f, \*args,**kwargs)\*\*

**arpes.utilities.xarray.lift\_datavar\_attrs(f)**

> Lifts a function that operates on a dictionary to a function that acts
> on the attributes of all the datavars in a xr.Dataset, as well as the
> Dataset attrs themselves. :param f: Function to apply :return:

**arpes.utilities.xarray.lift\_dataarray\_attrs(f)**

> Lifts a function that operates on a dictionary to a function that acts
> on the attributes of an xr.DataArray, producing a new xr.DataArray.
> Another option if you don’t need to create a new DataArray is to
> modify the attributes. :param f: :return: g: Function operating on the
> attributes of an xr.DataArray

**arpes.utilities.xarray.lift\_dataarray(f)**

> Lifts a function that operates on an np.ndarray’s values to one that
> acts on the values of an xr.DataArray :param f: :return: g: Function
> operating on an xr.DataArray
