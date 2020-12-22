arpes.plotting.qt\_ktool package
================================

Module contents
---------------

**class arpes.plotting.qt\_ktool.KTool(apply\_offsets=True,
zone=None,**kwargs)\*\*

> Bases:
> [arpes.utilities.qt.app.SimpleApp](arpes.utilities.qt.app#arpes.utilities.qt.app.SimpleApp)
>
> QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and
> PyQt5 for now we retain a number of the metaphors from BokehTool,
> including a “context” that stores the state, and can be used to
> programmatically interface with the tool
>
> `DEFAULT_COLORMAP = 'viridis'`
>
> `TITLE = 'KSpace-Tool'`
>
> `WINDOW_CLS`
>
> > alias of
> > [arpes.utilities.qt.windows.SimpleWindow](arpes.utilities.qt.windows#arpes.utilities.qt.windows.SimpleWindow)
>
> `WINDOW_SIZE = (5, 6)`
>
> **add\_contextual\_widgets()**
>
> **after\_show()**
>
> **before\_show()**
>
> **configure\_image\_widgets()**
>
> **layout()**
>
> **set\_data(data: Union\[xarray.core.dataarray.DataArray,
> xarray.core.dataset.Dataset\])**
>
> **update\_data()**
>
> **update\_offsets(offsets)**

**arpes.plotting.qt\_ktool.ktool(data:
Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\],**kwargs)\*\*
