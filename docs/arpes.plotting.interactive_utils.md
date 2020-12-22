arpes.plotting.interactive\_utils module
========================================

**class arpes.plotting.interactive\_utils.BokehInteractiveTool**

> Bases: `abc.ABC`
>
> `auto_rebin = True`
>
> `auto_zero_nans = True`
>
> **property debug\_div**
>
> **property default\_palette**
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
> **abstract tool\_handler(doc)**
>
> **update\_colormap\_for(plot\_name)**

**class arpes.plotting.interactive\_utils.CursorTool**

> Bases: `object`
>
> **add\_cursor\_lines(figure)**
>
> **property cursor**
>
> **property cursor\_dict**
>
> **property cursor\_dims**
