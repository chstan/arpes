# arpes.bootstrap module

**arpes.bootstrap.bootstrap(fn, skip=None, resample\_method=None)**

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

**arpes.bootstrap.resample\_true\_counts(data:
xarray.core.dataarray.DataArray) -\> xarray.core.dataarray.DataArray**

> Resamples histogrammed data where each count represents an actual
> electron. :param data: :return:

**arpes.bootstrap.bootstrap\_counts(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
N=1000, name=None) -\> xarray.core.dataset.Dataset**

> Parametric bootstrap for the number of counts in each detector channel
> for a time of flight/DLD detector, where each count represents an
> actual particle.
> 
> This function also introspects the data passed to determine whether
> there is a spin degree of freedom, and will bootstrap appropriately.
> 
> Currently we build all the samples at once instead of using a rolling
> algorithm. :param data: :return:

**arpes.bootstrap.bootstrap\_intensity\_polarization(data, N=100)**

> Uses the parametric bootstrap to get uncertainties on the intensity
> and polarization of ToF-SARPES data. :param data: :return:
