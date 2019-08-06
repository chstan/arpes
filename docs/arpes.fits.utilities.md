# arpes.fits.utilities module

The core of this module is *broadcast\_model* which is a serious
workhorse in PyARPES for analyses based on curve fitting. This allows
simple multidimensional curve fitting by iterative fitting across one or
many axes. Currently basic strategies are implemented, but in the future
we would like to provide:

1.    - Passing xr.DataArray values to parameter guesses and bounds,
        which  
        can be interpolated/selected to allow changing conditions
        throughout the curve fitting session.

2.    - A strategy allowing retries with initial guess taken from the  
        previous fit. This is similar to some adaptive curve fitting
        routines that have been proposed in the literature.

**arpes.fits.utilities.broadcast\_model(model\_cls: Union\[type,
List\[type\], Tuple\[type\]\], data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
broadcast\_dims, params=None, progress=True, dataset=True, weights=None,
safe=False, prefixes=None)**

> Perform a fit across a number of dimensions. Allows composite models
> as well as models defined and compiled through strings. :param
> model\_cls: :param data: :param broadcast\_dims: :param params: :param
> progress: :param dataset: :param weights: :param safe: :return:
