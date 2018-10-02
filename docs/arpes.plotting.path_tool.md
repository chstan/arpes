# arpes.plotting.path\_tool module

**class arpes.plotting.path\_tool.PathTool(**kwargs)\*\*

> Bases: `arpes.plotting.interactive_utils.SaveableTool`,
> [arpes.plotting.interactive\_utils.CursorTool](arpes.plotting.interactive_utils#arpes.plotting.interactive_utils.CursorTool)
> 
> Tool to allow drawing paths on data, creating selections based on
> paths, and masking regions around paths
> 
> Integrates with the tools in arpes.analysis.path
> 
> `auto_rebin = False`
> 
> `auto_zero_nans = False`
> 
> **deserialize(json\_data)**
> 
> **serialize()**
> 
> **tool\_handler(doc)**

**arpes.plotting.path\_tool.path\_tool(data:
Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\],**kwargs)\*\*
