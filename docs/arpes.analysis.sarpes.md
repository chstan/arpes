# arpes.analysis.sarpes module

**arpes.analysis.sarpes.to\_intensity\_polarization(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Converts from \[up, down\] representation (the spin projection) to
> \[intensity, polarization\] representation.
> 
> In this future, we should also make this also work with the timing
> signals. :param data: :return:

**arpes.analysis.sarpes.to\_up\_down(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Converts from \[intensity, polarization\] representation to \[up,
> down\] representation. :param data: :return:

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

> Smooths the up and down channels. :param data: :param args: :param
> kwargs: :return:
