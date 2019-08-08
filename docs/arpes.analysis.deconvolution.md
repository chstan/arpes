# arpes.analysis.deconvolution module

**arpes.analysis.deconvolution.deconvolve\_ice(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
psf, n\_iterations=5, deg=None)**

> Deconvolves data by a given point spread function using the iterative
> convolution extrapolation method. :param data: :param psf: :param
> n\_iterations – the number of convolutions to use for the fit (default
> 5): :param deg – the degree of the fitting polynominal (default
> n\_iterations-3): :return DataArray or numpy.ndarray – based on input
> type:

**arpes.analysis.deconvolution.deconvolve\_rl(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
psf=None, n\_iterations=10, axis=None, sigma=None, mode='reflect',
progress=True)**

> Deconvolves data by a given point spread function using the
> Richardson-Lucy method.
> 
>   - Parameters
>     
>       - **data** –
>     
>       - **-- for 1d,if not specified,must specify axis and sigma**
>         (*psf*) –
>     
>       -   - **-- the number of convolutions to use for the fit**\*\*  
>             (\***\*default 50)** (*n\_iterations*) –
>     
>       - **axis** –
>     
>       - **sigma** –
>     
>       - **mode** –
>     
>       - **progress** –
> 
>   - Return DataArray or numpy.ndarray – based on input type

**arpes.analysis.deconvolution.make\_psf1d(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
dim, sigma)**

> Produces a 1-dimensional gaussian point spread function for use in
> deconvolve\_rl.
> 
>   - Parameters
>     
>       - **data** –
>       - **dim** –
>       - **sigma** –
> 
>   - Return DataArray
