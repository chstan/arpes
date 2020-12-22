arpes.plotting.qt\_tool package
===============================

Submodules
----------

-   [arpes.plotting.qt\_tool.AxisInfoWidget
    module](arpes.plotting.qt_tool.AxisInfoWidget)
-   [arpes.plotting.qt\_tool.BinningInfoWidget
    module](arpes.plotting.qt_tool.BinningInfoWidget)

Module contents
---------------

**class arpes.plotting.qt\_tool.QtTool**

> Bases:
> [arpes.utilities.qt.app.SimpleApp](arpes.utilities.qt.app#arpes.utilities.qt.app.SimpleApp)
>
> QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and
> PyQt5 for now we retain a number of the metaphors from BokehTool,
> including a “context” that stores the state, and can be used to
> programmatically interface with the tool
>
> `TITLE = 'Qt Tool'`
>
> `WINDOW_CLS`
>
> > alias of `arpes.plotting.qt_tool.QtToolWindow`
>
> `WINDOW_SIZE = (5, 5)`
>
> **add\_contextual\_widgets()**
>
> **after\_show()**
>
> **before\_show()**
>
> **property binning**
>
> **center\_cursor()**
>
> **configure\_image\_widgets()**
>
> **connect\_cursor(dimension, the\_line)**
>
> **construct\_axes\_tab()**
>
> **construct\_binning\_tab()**
>
> **construct\_kspace\_tab()**
>
> **layout()**
>
> **scroll(delta)**
>
> **set\_data(data: Union\[xarray.core.dataarray.DataArray,
> xarray.core.dataset.Dataset\])**
>
> **transpose(transpose\_order)**
>
> **transpose\_to\_front(dim)**
>
> **update\_cursor\_position(new\_cursor, force=False,
> keep\_levels=True)**

**arpes.plotting.qt\_tool.qt\_tool(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**
