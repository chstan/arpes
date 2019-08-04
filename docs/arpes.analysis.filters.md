# arpes.analysis.filters module

**arpes.analysis.filters.gaussian\_filter\_arr(arr:
xarray.core.dataarray.DataArray, sigma=None, n=1, default\_size=1)**

> Functionally wraps scipy.ndimage.filters.gaussian\_filter with the
> advantage that the sigma is coordinate aware.
> 
>   - Parameters
>     
>       - **arr** –
>     
>       -   - **sigma** – Kernel sigma, specified in terms of axis
>             units.  
>             An axis that is not specified will have a kernel width of
>             *default\_size* in index units.
>     
>       - **n** – Repeats n times.
>     
>       -   - **default\_size** – Changes the default kernel width for
>             axes  
>             not specified in *sigma*. Changing this parameter and
>             leaving *sigma* as None allows you to smooth with an
>             even-width kernel in index-coordinates.
> 
>   - Returns  
>     xr.DataArray: smoothed data.

**arpes.analysis.filters.gaussian\_filter(sigma=None, n=1)**

> A partial application of *gaussian\_filter\_arr* that can be passed to
> derivative analysis functions. :param sigma: :param n: :return:

**arpes.analysis.filters.boxcar\_filter\_arr(arr:
xarray.core.dataarray.DataArray, size=None, n=1, default\_size=1,
skip\_nan=True)**

> Functionally wraps scipy.ndimage.filters.gaussian\_filter with the
> advantage that the sigma is coordinate aware.
> 
>   - Parameters
>     
>       - **arr** –
>     
>       -   - **size** – Kernel size, specified in terms of axis units.
>             An  
>             axis that is not specified will have a kernel width of
>             *default\_size* in index units.
>     
>       - **n** – Repeats n times.
>     
>       -   - **default\_size** – Changes the default kernel width for
>             axes  
>             not specified in *sigma*. Changing this parameter and
>             leaving *sigma* as None allows you to smooth with an
>             even-width kernel in index-coordinates.
>     
>       -   - **skip\_nan** – By default, masks parts of the data which
>             are  
>             NaN to prevent poor filter results.
> 
>   - Returns  
>     xr.DataArray: smoothed data.

**arpes.analysis.filters.boxcar\_filter(size=None, n=1)**

> A partial application of *boxcar\_filter\_arr* that can be passed to
> derivative analysis functions. :param size: :param n: :return:
