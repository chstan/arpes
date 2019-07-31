# arpes.plotting.interactive\_utils module

**class arpes.plotting.interactive\_utils.BokehInteractiveTool**

> Bases: `abc.ABC`
> 
> `auto_rebin = True`
> 
> `auto_zero_nans = True`
> 
> `debug_div`
> 
> `default_palette`
> 
> **init\_bokeh\_server()**
> 
> **load\_settings(**kwargs)\*\*
> 
> **make\_tool(arr: Union\[xarray.core.dataarray.DataArray, str\],
> notebook\_url=None, notebook\_handle=True,**kwargs)\*\*
> 
> `rebin_size = 800`
> 
> **tool\_handler(doc)**
> 
> **update\_colormap\_for(plot\_name)**

**class arpes.plotting.interactive\_utils.CursorTool**

> Bases: `object`
> 
> **add\_cursor\_lines(figure)**
> 
> `cursor`
> 
> `cursor_dict`
> 
> `cursor_dims`
