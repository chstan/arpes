# arpes.plotting.utils module

**arpes.plotting.utils.path\_for\_plot(desired\_path)**

**arpes.plotting.utils.path\_for\_holoviews(desired\_path)**

**arpes.plotting.utils.name\_for\_dim(dim\_name, escaped=True)**

**arpes.plotting.utils.label\_for\_colorbar(data)**

**arpes.plotting.utils.label\_for\_dim(data=None, dim\_name=None,
escaped=True)**

**arpes.plotting.utils.label\_for\_symmetry\_point(point\_name)**

**arpes.plotting.utils.savefig(desired\_path, dpi=400,**kwargs)\*\*

**class arpes.plotting.utils.AnchoredHScaleBar(size=1, extent=0.03,
label='', loc=2, ax=None, pad=0.4, borderpad=0.5, ppad=0, sep=2,
prop=None, label\_color=None, frameon=True,**kwargs)\*\*

> Bases: `matplotlib.offsetbox.AnchoredOffsetbox`
> 
> size: length of bar in data units extent : height of bar ends in axes
> units

**arpes.plotting.utils.calculate\_aspect\_ratio(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

**arpes.plotting.utils.unit\_for\_dim(dim\_name, escaped=True)**

**arpes.plotting.utils.polarization\_colorbar(ax=None)**

**arpes.plotting.utils.temperature\_colormap(high=300, low=0,
cmap=None)**

**arpes.plotting.utils.temperature\_colormap\_around(central,
range=50)**

**arpes.plotting.utils.temperature\_colorbar(high=300, low=0, ax=None,
cmap=None,**kwargs)\*\*

**arpes.plotting.utils.temperature\_colorbar\_around(central, range=50,
ax=None,**kwargs)\*\*

**arpes.plotting.utils.dos\_axes(orientation='horiz', figsize=None,
with\_cbar=True)**

> Orientation option should be ‘horiz’ or ‘vert’ :param orientation:
> :param figsize: :return:

**arpes.plotting.utils.fancy\_labels(ax\_or\_ax\_set, data=None)**

**arpes.plotting.utils.inset\_cut\_locator(data, reference\_data=None,
ax=None, location=None, color=None,**kwargs)\*\*

> Plots a reference cut location :param data: :param reference\_data:
> :param ax: :param location: :param kwargs: :return:
