# arpes.analysis.resolution module

Contains calibrations and information for spectrometer resolution.

**arpes.analysis.resolution.analyzer\_resolution(analyzer\_information,
slit\_width=None, slit\_number=None, pass\_energy=10)**

**arpes.analysis.resolution.analyzer\_resolution\_estimate(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> For hemispherical analyzers, this can be determined by the slit and
> pass energy settings.
> 
> Roughly, :param data: :return:

**arpes.analysis.resolution.beamline\_resolution\_estimate(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

**arpes.analysis.resolution.energy\_resolution\_from\_exit\_slit(table,
photon\_energy, exit\_slit\_size)**

> Assumes an exact match on the photon energy, though that interpolation
> could also be pulled into here… :param table: :param photon\_energy:
> :param exit\_slit\_size: :return:

**arpes.analysis.resolution.r8000(slits)**

**arpes.analysis.resolution.thermal\_broadening\_estimate(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Simple Fermi-Dirac broadening :param data: :return:

**arpes.analysis.resolution.total\_resolution\_estimate(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
include\_thermal\_broadening=False)**

> Gives the quadrature sum estimate of the resolution of an ARPES
> spectrum that is decorated with appropriate information.
> 
> For synchrotron ARPES, this typically means the scan has the photon
> energy, exit slit information and analyzer slit settings :return:
