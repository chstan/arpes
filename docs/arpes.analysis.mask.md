# arpes.analysis.mask module

**arpes.analysis.mask.polys\_to\_mask(mask\_dict, coords, shape,
radius=None, invert=False)**

> Converts a mask definition in terms of the underlying polygon to a
> True/False mask array using the coordinates and shape of the target
> data.
> 
> This process “specializes” a mask to a particular shape, whereas masks
> given by polygon definitions are general to any data with appropriate
> dimensions, because waypoints are given in unitful values rather than
> index values. :param mask\_dict: :param coords: :param shape: :param
> radius: :param invert: :return:

**arpes.analysis.mask.apply\_mask(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
mask, replace=nan, radius=None, invert=False)**

> Applies a logical mask, i.e. one given in terms of polygons, to a
> specific piece of data. This can be used to set values outside or
> inside a series of polygon masks to a given value or to NaN.
> 
> Expanding or contracting the masked region can be accomplished with
> the radius argument, but by default strict inclusion is used.
> 
> Some masks include a *fermi* parameter which allows for clipping the
> detector boundaries in a semi-automated fashion. If this is included,
> only 200meV above the Fermi level will be included in the returned
> data. This helps to prevent very large and undesirable regions filled
> with only the replacement value which can complicate automated
> analyses that rely on masking.
> 
>   - Parameters
>     
>       - **data** – Data to mask.
>     
>       -   - **mask** – Logical definition of the mask, appropriate
>             for  
>             passing to *polys\_to\_mask*
>     
>       - **replace** – The value to substitute for pixels masked.
>     
>       - **radius** – Radius by which to expand the masked area.
>     
>       -   - **invert** – Allows logical inversion of the masked parts
>             of  
>             the data. By default, the area inside the polygon sequence
>             is replaced by *replace*.
> 
>   - Returns

**arpes.analysis.mask.raw\_poly\_to\_mask(poly)**

> There’s not currently much metadata attached to masks, but this is
> around if we ever decide that we need to implement more complicated
> masking schemes.
> 
> In particular, we might want to store also whether the interior or
> exterior is the masked region, but this is functionally achieved for
> now with the *invert* flag in other functions.
> 
>   - Parameters  
>     **poly** – Polygon implementing a masked region.
> 
>   - Returns

**arpes.analysis.mask.apply\_mask\_to\_coords(data:
xarray.core.dataset.Dataset, mask, dims, invert=True)**
