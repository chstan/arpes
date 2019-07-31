# arpes.analysis.resolution module

Contains calibrations and information for spectrometer resolution.

**arpes.analysis.resolution.total\_resolution\_estimate(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
include\_thermal\_broadening=False, meV=False)**

> Gives the quadrature sum estimate of the resolution of an ARPES
> spectrum that is decorated with appropriate information.
> 
> For synchrotron ARPES, this typically means the scan has the photon
> energy, exit slit information and analyzer slit settings :return:
