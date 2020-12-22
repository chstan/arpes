arpes.plotting.utils module
===========================

**class arpes.plotting.utils.AnchoredHScaleBar(size=1, extent=0.03,
label='', loc=2, ax=None, pad=0.4, borderpad=0.5, ppad=0, sep=2,
prop=None, label\_color=None, frameon=True,**kwargs)\*\*

> Bases: `matplotlib.offsetbox.AnchoredOffsetbox`
>
> Modified from [this StackOverflow
> question](https://stackoverflow.com/questions/43258638/) as alternate
> to the one provided through matplotlib
>
> size: length of bar in data units extent : height of bar ends in axes
> units

**arpes.plotting.utils.axis\_to\_data\_units(points, ax=None)**

**arpes.plotting.utils.calculate\_aspect\_ratio(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

**arpes.plotting.utils.dark\_background(overrides)**

**arpes.plotting.utils.data\_to\_axis\_units(points, ax=None)**

**arpes.plotting.utils.daxis\_ddata\_units(ax=None)**

**arpes.plotting.utils.ddata\_daxis\_units(ax=None)**

**arpes.plotting.utils.dos\_axes(orientation='horiz', figsize=None,
with\_cbar=True)**

> Orientation option should be ‘horiz’ or ‘vert’
>
> Parameters  
> -   **orientation** –
> -   **figsize** –
> -   **with\_cbar** –
>
> Returns  

**arpes.plotting.utils.fancy\_labels(ax\_or\_ax\_set, data=None)**

> Attaches better display axis labels for all axes that can be traversed
> in the passed figure or axes.
>
> Parameters  
> -   **ax\_or\_ax\_set** –
> -   **data** –
>
> Returns  

**arpes.plotting.utils.frame\_with(ax, color='red', linewidth=2)**

**arpes.plotting.utils.generic\_colorbarmap\_for\_data(data:
xarray.core.dataarray.DataArray, keep\_ticks=True, ax=None,**kwargs)\*\*

**arpes.plotting.utils.get\_colorbars(fig=None)**

**arpes.plotting.utils.h\_gradient\_fill(x1, x2, x\_solid,
fill\_color=None, ax=None, zorder=None, alpha=None,**kwargs)\*\*

> Fills a gradient between x1 and x2. If x\_solid is not None, the
> gradient will be extended at the maximum opacity from the closer limit
> towards x\_solid.
>
> Parameters  
> -   **x1** –
> -   **x2** –
> -   **x\_solid** –
> -   **fill\_color** –
> -   **ax** –
> -   **zorder** –
> -   **alpha** –
> -   **kwargs** –
>
> Returns  

**arpes.plotting.utils.imshow\_arr(arr, ax=None, over=None,
origin='lower', aspect='auto', alpha=None, vmin=None, vmax=None,
cmap=None,**kwargs)\*\*

> Similar to plt.imshow but users different default origin, and sets
> appropriate extent on the plotted data. :param arr: :param ax:
> :return:

**arpes.plotting.utils.imshow\_mask(mask, ax=None, over=None,
cmap=None,**kwargs)\*\*

**arpes.plotting.utils.inset\_cut\_locator(data, reference\_data=None,
ax=None, location=None, color=None,**kwargs)\*\*

> Plots a reference cut location :param data: :param reference\_data:
> :param ax: :param location: :param kwargs: :return:

**arpes.plotting.utils.invisible\_axes(ax)**

**arpes.plotting.utils.label\_for\_colorbar(data)**

**arpes.plotting.utils.label\_for\_dim(data=None, dim\_name=None,
escaped=True)**

**arpes.plotting.utils.label\_for\_symmetry\_point(point\_name)**

**arpes.plotting.utils.lineplot\_arr(arr, ax=None, method='plot',
mask=None, mask\_kwargs=None,**kwargs)\*\*

**arpes.plotting.utils.load\_data\_for\_figure(p: Union\[str,
pathlib.Path\])**

**arpes.plotting.utils.mean\_annotation(eV=None, phi=None)**

**arpes.plotting.utils.mod\_plot\_to\_ax(data, ax, mod,**kwargs)\*\*

> Plots a model onto an axis using the data range from the passed data

**arpes.plotting.utils.name\_for\_dim(dim\_name, escaped=True)**

**arpes.plotting.utils.no\_ticks(ax)**

**arpes.plotting.utils.path\_for\_holoviews(desired\_path)**

**arpes.plotting.utils.path\_for\_plot(desired\_path)**

**arpes.plotting.utils.plot\_arr(arr=None, ax=None, over=None,
mask=None,**kwargs)\*\*

**arpes.plotting.utils.polarization\_colorbar(ax=None)**

**arpes.plotting.utils.quick\_tex(latex\_fragment, ax=None,
fontsize=30)**

> Sometimes you just need to render some latex and getting a latex
> session running is far too much effort. :param latex\_fragment:
> :return:

**arpes.plotting.utils.remove\_colorbars(fig=None)**

> Removes colorbars from given (or, if no given figure, current)
> matplotlib figure.
>
> Parameters  
> **(default plt.gcf())** (*fig*) –

**arpes.plotting.utils.savefig(desired\_path, dpi=400, data=None,
save\_data=None, paper=False,**kwargs)\*\*

**arpes.plotting.utils.simple\_ax\_grid(n\_axes,
figsize=None,**kwargs)\*\*

> Generates a square-ish set of axes and hides the extra ones
>
> It would be nice to accept an “aspect ratio” item that will attempt to
> fix the grid dimensions to get an aspect ratio close to the desired
> one.
>
> Parameters  
> -   **n\_axes** –
> -   **figsize** –
> -   **kwargs** –
>
> Returns  

**arpes.plotting.utils.sum\_annotation(eV=None, phi=None)**

**arpes.plotting.utils.summarize(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
axes=None)**

**arpes.plotting.utils.swap\_axis\_sides(ax)**

**arpes.plotting.utils.swap\_xaxis\_side(ax)**

**arpes.plotting.utils.swap\_yaxis\_side(ax)**

**arpes.plotting.utils.temperature\_colorbar(high=300, low=0, ax=None,
cmap=None,**kwargs)\*\*

**arpes.plotting.utils.temperature\_colorbar\_around(central, range=50,
ax=None,**kwargs)\*\*

**arpes.plotting.utils.temperature\_colormap(high=300, low=0,
cmap=None)**

**arpes.plotting.utils.temperature\_colormap\_around(central,
range=50)**

**arpes.plotting.utils.transform\_labels(transform\_fn, fig=None,
include\_titles=True)**

**arpes.plotting.utils.unchanged\_limits(ax)**

> Context manager that retains axis limits

**arpes.plotting.utils.unit\_for\_dim(dim\_name, escaped=True)**

**arpes.plotting.utils.v\_gradient\_fill(y1, y2, y\_solid,
fill\_color=None, ax=None, zorder=None, alpha=None,**kwargs)\*\*

> Fills a gradient vertically between y1 and y2. If y\_solid is not
> None, the gradient will be extended at the maximum opacity from the
> closer limit towards y\_solid.
>
> Parameters  
> -   **y1** –
> -   **y2** –
> -   **y\_solid** –
> -   **fill\_color** –
> -   **ax** –
> -   **zorder** –
> -   **alpha** –
> -   **kwargs** –
>
> Returns  
