# arpes.plotting.spatial module

**arpes.plotting.spatial.reference\_scan\_spatial(data,
out=None,**kwargs)\*\*

**arpes.plotting.spatial.plot\_spatial\_reference(reference\_map:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
data\_list: List\[Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\]\], offset\_list:
Optional\[List\[Dict\[str, Any\]\]\] = None, annotation\_list:
Optional\[List\[str\]\] = None, out: Optional\[str\] = None, plot\_refs:
bool = True)**

> Helpfully plots data against a reference scanning dataset. This is
> essential to understand where data was taken and can be used early in
> the analysis phase in order to highlight the location of your datasets
> against core levels, etc.
> 
>   - Parameters
>     
>       - **reference\_map** – A scanning photoemission like dataset
>     
>       -   - **data\_list** – A list of datasets you want to plot the  
>             relative locations of
>     
>       -   - **offset\_list** – Optionally, offsets given as
>             coordinate  
>             dicts
>     
>       -   - **annotation\_list** – Optionally, text annotations for
>             the  
>             data
>     
>       -   - **out** – Where to save the figure if we are outputting
>             to  
>             disk
>     
>       -   - **plot\_refs** – Whether to plot reference figures for
>             each of  
>             the pieces of data in *data\_list*
> 
>   - Returns
