# arpes.bootstrap module

**arpes.bootstrap.bootstrap(fn, skip=None)**

**arpes.bootstrap.estimate\_prior\_adjustment(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
region: Union\[dict, str\] = None)**

> Estimates the parameters of a distribution generating the intensity
> histogram of pixels in a spectrum. In a perfectly linear,
> single-electron single-count detector, this would be a poisson
> distribution with lambda=mean(counts) over the window. Despite this,
> we can estimate lambda phenomenologically and verify that a Poisson
> distribution provides a good prior for the data, allowing us to
> perform statistical bootstrapping.
> 
> You should use this with a spectrum that has uniform intensity, i.e.
> with a copper reference or similar.
> 
>   - Parameters  
>     **data** –
> 
>   - Returns  
>     returns sigma / mu, adjustment factor for the Poisson distribution
