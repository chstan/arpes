# arpes.analysis.path module

**arpes.analysis.path.discretize\_path(path:
xarray.core.dataset.Dataset, n\_points=None, scaling=None)**

> Shares logic with slice\_along\_path :param path: :param n\_points:
> :return:

**arpes.analysis.path.select\_along\_path(path:
xarray.core.dataset.Dataset, data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
radius=None, n\_points=None, fast=True, scaling=None,**kwargs)\*\*

> Performs integration along a path. This functionally allows for
> performing a finite width cut (with finite width perpendicular to the
> local path direction) along some path, and integrating along this
> perpendicular selection. This allows for better statistics in
> oversampled data.
> 
>   - Parameters
>     
>       - **path** –
>     
>       - **data** –
>     
>       -   - **radius** – A number or dictionary of radii to use for
>             the  
>             selection along different dimensions, if none is provided
> 
> reasonable values will be chosen. Alternatively, you can pass radii
> via *{dim}\_r* kwargs as well, i.e. ‘eV\_r’ or ‘kp\_r’ :param
> n\_points: The number of points to interpolate along the path, by
> default we will infer a reasonable number from the radius parameter,
> if provided or inferred :param fast: If fast is true, will use
> rectangular selections rather than ellipsoid ones :return:

**arpes.analysis.path.path\_from\_points(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
symmetry\_points\_or\_interpolation\_points)**

> Acceepts a list of either tuples or point references. Point references
> can be string keys to *.attrs\[‘symmetry\_points’\]* This is the same
> behavior as *analysis.slice\_along\_path* and underlies the logic
> there. :param data: :param
> symmetry\_points\_or\_interpolation\_points: :return:
