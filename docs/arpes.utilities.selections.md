arpes.utilities.selections module
=================================

This package contains utilities related to taking more complicated
shaped selections around data.

Currently it houses just utilities for forming disk and annular
selections out of data.

**arpes.utilities.selections.ravel\_from\_mask(data, mask)**

> Selects out the data from a ND array whose points are marked true in
> *mask*. See also *unravel\_from\_mask* below which allows you to write
> back into data after you have transformed the 1D output in some way.
>
> These two functions are especially useful for hierarchical curve
> fitting where you want to rerun a fit over a subset of the data with a
> different model, such as when you know some of the data is best
> described by two bands rather than one. :param data: :param mask:
> :return:

**arpes.utilities.selections.select\_disk(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
radius, outer\_radius=None, around: Optional\[Union\[Dict,
xarray.core.dataset.Dataset\]\] = None, invert=False,**kwargs) -&gt;
Tuple\[Dict\[str, numpy.ndarray\], numpy.ndarray, numpy.ndarray\]\*\*

> Selects the data in a disk (or annulus if *outer\_radius* is provided)
> around the point described by *around* and *kwargs*. A point is a
> labeled collection of coordinates that matches all of the dimensions
> of *data*. The coordinates can either be passed through a dict as
> *around*, as the coordinates of a Dataset through *around* or
> explicitly in keyword argument syntax through *kwargs*. The radius for
> the disk is specified through the required *radius* parameter.
>
> Data is returned as a tuple with the type Tuple\[Dict\[str,
> np.ndarray\], np.ndarray, np.ndarray\] containing a dictionary with
> the filtered lists of coordinates, an array with the original data
> values at these coordinates, and finally an array of the distances to
> the requested point. :param data: :param radius: :param outer\_radius:
> :param around: :param invert: :param kwargs: :return:

**arpes.utilities.selections.select\_disk\_mask(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
radius, outer\_radius=None, around: Optional\[Union\[Dict,
xarray.core.dataset.Dataset\]\] = None, flat=False,**kwargs) -&gt;
numpy.ndarray\*\*

> A complement to select\_disk which only generates the mask.
>
> Selects the data in a disk around the point described by *around* and
> *kwargs*. A point is a labelled collection of coordinates that matches
> all of the dimensions of *data*. The coordinates can either be passed
> through a dict as *around*, as the coordinates of a Dataset through
> *around* or explicitly in keyword argument syntax through *kwargs*.
> The radius for the disk is specified through the required *radius*
> parameter.
>
> Returns the ND mask that represents the filtered coordinates. :param
> data: :param around: :param flat: Whether to return the mask as a 1D
> (raveled) mask (flat=True) or as a ND mask with the
>
> Parameters  
> **kwargs** –
>
> Returns  

**arpes.utilities.selections.unravel\_from\_mask(template, mask, values,
default=nan)**

> Creates an array :param template: :param mask: :param values: :param
> default: :return:
