# arpes.analysis.sarpes module

**arpes.analysis.sarpes.to\_intensity\_polarization(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Converts to intensity and polarization, rather than the spin
> components.
> 
> TODO, make this also work with the timing signals :param data:
> :return:

**arpes.analysis.sarpes.to\_up\_down(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

**arpes.analysis.sarpes.normalize\_sarpes\_photocurrent(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Normalizes the down channel so that it matches the up channel in terms
> of mean photocurrent. Destroys the integrity of “count” data.
> 
>   - Parameters  
>     **data** –
> 
>   - Returns

**arpes.analysis.sarpes.sarpes\_smooth(data:
xarray.core.dataset.Dataset, \*args,**kwargs)\*\*
