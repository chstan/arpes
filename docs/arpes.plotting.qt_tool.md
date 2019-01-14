# arpes.plotting.qt\_tool module

**class arpes.plotting.qt\_tool.QtTool**

> Bases: `object`
> 
> QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and
> PyQt5 for now we retain a number of the metaphors from BokehTool,
> including a “context” that stores the state, and can be used to
> programmatically interface with the tool
> 
> **configure\_image\_widgets()**
> 
> **connect\_cursor(dimension, the\_line)**
> 
> **generate\_marginal\_for(dimensions, column, row, name=None,
> orientation='horiz', cursors=False)**
> 
> **set\_colormap(colormap)**
> 
> **set\_data(data: Union\[xarray.core.dataarray.DataArray,
> xarray.core.dataset.Dataset\])**
> 
> **start()**
> 
> **update\_cursor\_position(new\_cursor, force=False)**

**arpes.plotting.qt\_tool.qt\_tool(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**
