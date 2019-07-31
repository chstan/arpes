# arpes.plotting.tof module

This module is a bit of a misnomer, in that it also applies perfectly
well to data collected by a delay line on a hemisphere, the important
point is that the data in any given channel should correspond to the
true number of electrons that arrived in that channel.

Plotting routines here are ones that include statistical errorbars.
Generally for datasets in PyARPES, an xr.Dataset will hold the standard
deviation data for a given variable on *{var\_name}\_std*.

**arpes.plotting.tof.plot\_with\_std(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
name\_to\_plot=None, ax=None, out=None,**kwargs)\*\*

**arpes.plotting.tof.scatter\_with\_std(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
name\_to\_plot=None, ax=None, fmt='o', out=None,**kwargs)\*\*
