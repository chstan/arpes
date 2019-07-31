# arpes.plotting.qt\_tool package

## Submodules

  - [arpes.plotting.qt\_tool.AxisInfoWidget
    module](arpes.plotting.qt_tool.AxisInfoWidget)
  - [arpes.plotting.qt\_tool.BinningInfoWidget
    module](arpes.plotting.qt_tool.BinningInfoWidget)
  - [arpes.plotting.qt\_tool.DataArrayImageView
    module](arpes.plotting.qt_tool.DataArrayImageView)
  - [arpes.plotting.qt\_tool.HelpDialog
    module](arpes.plotting.qt_tool.HelpDialog)
  - [arpes.plotting.qt\_tool.utils module](arpes.plotting.qt_tool.utils)

## Module contents

**class arpes.plotting.qt\_tool.QtTool**

> Bases: `object`
> 
> QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and
> PyQt5 for now we retain a number of the metaphors from BokehTool,
> including a “context” that stores the state, and can be used to
> programmatically interface with the tool
> 
> **add\_contextual\_widgets()**
> 
> `binning`
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
> **generate\_marginal\_for(dimensions, column, row, name=None,
> orientation='horiz', cursors=False)**
> 
> `info_tab`
> 
> **print(\*args,**kwargs)\*\*
> 
> **set\_colormap(colormap)**
> 
> **set\_data(data: Union\[xarray.core.dataarray.DataArray,
> xarray.core.dataset.Dataset\])**
> 
> **start()**
> 
> **transpose(transpose\_order)**
> 
> **transpose\_to\_front(dim)**
> 
> **update\_cursor\_position(new\_cursor, force=False)**

**arpes.plotting.qt\_tool.qt\_tool(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**
