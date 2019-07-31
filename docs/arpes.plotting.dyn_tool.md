# arpes.plotting.dyn\_tool module

**class arpes.plotting.dyn\_tool.DynamicTool(analysis\_fn,
widget\_specification,**kwargs)\*\*

> Bases:
> [arpes.plotting.interactive\_utils.BokehInteractiveTool](arpes.plotting.interactive_utils#arpes.plotting.interactive_utils.BokehInteractiveTool),
> [arpes.plotting.interactive\_utils.CursorTool](arpes.plotting.interactive_utils#arpes.plotting.interactive_utils.CursorTool)
> 
> Presents a utility to rerun a function with different arguments and
> see the result of the function
> 
> **tool\_handler(doc)**

**arpes.plotting.dyn\_tool.dyn(dynamic\_function: Callable, data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
widget\_specifications=None)**
