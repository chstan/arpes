# arpes.analysis.tarpes module

**arpes.analysis.tarpes.find\_t0(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
e\_bound=0.02, approx=True)**

> Attempts to find the effective t0 in a spectrum by fitting a peak to
> the counts that occur far enough above e\_F :param data: :param
> e\_bound: Lower bound on the energy to use for the fitting :return:

**arpes.analysis.tarpes.relative\_change(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
t0=None, buffer=0.3, normalize\_delay=True)**

> Like normalized\_relative\_change, but only subtracts the before t0
> data rather than normalizing by the original spectrum’s intensity in
> each frame. :param data: :param t0: :param buffer: :param
> normalize\_delay: :return:

**arpes.analysis.tarpes.normalized\_relative\_change(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
t0=None, buffer=0.3, normalize\_delay=True)**

> Calculates a normalized relative change, obtained by normalizing along
> the pump-probe “delay” axis and then subtracting the mean before t0
> data and dividing by the original spectrum. :param data: :param t0:
> :param buffer: :param normalize\_delay: :return:
