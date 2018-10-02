# arpes.provenance module

**arpes.provenance.attach\_id(data)**

**arpes.provenance.provenance(child\_arr:
xarray.core.dataarray.DataArray, parent\_arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset,
List\[Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\]\]\], record, keep\_parent\_ref=False)**

**arpes.provenance.provenance\_from\_file(child\_arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
file, record)**

**arpes.provenance.provenance\_multiple\_parents(child\_arr: (\<class
'xarray.core.dataarray.DataArray'\>, \<class
'xarray.core.dataset.Dataset'\>), parents, record,
keep\_parent\_ref=False)**

**arpes.provenance.save\_plot\_provenance(plot\_fn)**

> A decorator that automates saving the provenance information for a
> particular plot. A plotting function creates an image or movie
> resource at some location on the filesystem.
> 
> In order to hook into this decorator appropriately, because there is
> no way that I know of of temporarily overriding the behavior of the
> open builtin in order to monitor for a write.
> 
>   - Parameters  
>     **plot\_fn** – A plotting function to decorate
> 
>   - Returns

**arpes.provenance.update\_provenance(what, record\_args=None,
keep\_parent\_ref=False)**

> Provides a decorator that promotes a function to one that records data
> provenance.
> 
>   - Parameters
>     
>       -   - **what** – Description of what transpired, to put into
>             the  
>             record.
>     
>       -   - **record\_args** – Unused presently, will allow recording
>             args  
>             into record.
>     
>       -   - **keep\_parent\_ref** – Whether to keep a pointer to the  
>             parents in the hierarchy or not.
> 
>   - Returns  
>     decorator
