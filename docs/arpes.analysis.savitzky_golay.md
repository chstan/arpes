# arpes.analysis.savitzky\_golay module

**arpes.analysis.savitzky\_golay.savitzky\_golay(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset,
list, numpy.ndarray\], window\_size, order, deriv=0, rate=1, dim=None)**

> Implements a Savitzky Golay filter with given window size. You can
> specify “pass through” dimensions which will not be touched with the
> *dim* argument. This allows for filtering each frame of a map or each
> equal-energy contour in a 3D dataset, for instance.
> 
>   - Parameters
>     
>       - **data** – Input data.
>     
>       -   - **window\_size** – Number of points in the window that
>             the  
>             filter uses locally.
>     
>       - **order** – The polynomial order used in the convolution.
>     
>       - **deriv** –
>     
>       - **rate** –
>     
>       - **dim** –
> 
>   - Returns
