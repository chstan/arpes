# arpes.utilities.conversion.bounds\_calculations module

**arpes.utilities.conversion.bounds\_calculations.calculate\_kp\_kz\_bounds(arr:
xarray.core.dataarray.DataArray)**

**arpes.utilities.conversion.bounds\_calculations.calculate\_kx\_ky\_bounds(arr:
xarray.core.dataarray.DataArray)**

> Calculates the kx and ky range for a dataset with a fixed photon
> energy
> 
> This is used to infer the gridding that should be used for a k-space
> conversion. Based on Jonathan Denlinger’s old codes :param arr:
> Dataset that includes a key indicating the photon energy of the scan
> :return: ((kx\_low, kx\_high,), (ky\_low,
ky\_high,))

**arpes.utilities.conversion.bounds\_calculations.calculate\_kp\_bounds(arr:
xarray.core.dataarray.DataArray)**
