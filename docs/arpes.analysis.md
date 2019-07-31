# arpes.analysis package

## Submodules

  - [arpes.analysis.band\_analysis module](arpes.analysis.band_analysis)
  - [arpes.analysis.band\_analysis\_utils
    module](arpes.analysis.band_analysis_utils)
  - [arpes.analysis.decomposition module](arpes.analysis.decomposition)
  - [arpes.analysis.deconvolution module](arpes.analysis.deconvolution)
  - [arpes.analysis.derivative module](arpes.analysis.derivative)
  - [arpes.analysis.fft module](arpes.analysis.fft)
  - [arpes.analysis.filters module](arpes.analysis.filters)
  - [arpes.analysis.fs module](arpes.analysis.fs)
  - [arpes.analysis.gap module](arpes.analysis.gap)
  - [arpes.analysis.general module](arpes.analysis.general)
  - [arpes.analysis.kfermi module](arpes.analysis.kfermi)
  - [arpes.analysis.mask module](arpes.analysis.mask)
  - [arpes.analysis.path module](arpes.analysis.path)
  - [arpes.analysis.pocket module](arpes.analysis.pocket)
  - [arpes.analysis.resolution module](arpes.analysis.resolution)
  - [arpes.analysis.sarpes module](arpes.analysis.sarpes)
  - [arpes.analysis.savitzky\_golay
    module](arpes.analysis.savitzky_golay)
  - [arpes.analysis.shirley module](arpes.analysis.shirley)
  - [arpes.analysis.statistics module](arpes.analysis.statistics)
  - [arpes.analysis.tarpes module](arpes.analysis.tarpes)
  - [arpes.analysis.xps module](arpes.analysis.xps)

## Module contents

**arpes.analysis.d1\_along\_axis(arr: xarray.core.dataarray.DataArray,
axis=None, smooth\_fn=None, \*, order=1)**

> Like curvature, performs a second derivative. You can pass a function
> to use for smoothing through the parameter smooth\_fn, otherwise no
> smoothing will be performed.
> 
> You can specify the axis to take the derivative along with the axis
> param, which expects a string. If no axis is provided the axis will be
> chosen from among the available ones according to the preference for
> axes here, the first available being taken:
> 
> \[‘eV’, ‘kp’, ‘kx’, ‘kz’, ‘ky’, ‘phi’, ‘beta’, ‘theta\] :param arr:
> :param axis: :param smooth\_fn: :param order: Specifies how many
> derivatives to take :return:

**arpes.analysis.d2\_along\_axis(arr: xarray.core.dataarray.DataArray,
axis=None, smooth\_fn=None, \*, order=2)**

> Like curvature, performs a second derivative. You can pass a function
> to use for smoothing through the parameter smooth\_fn, otherwise no
> smoothing will be performed.
> 
> You can specify the axis to take the derivative along with the axis
> param, which expects a string. If no axis is provided the axis will be
> chosen from among the available ones according to the preference for
> axes here, the first available being taken:
> 
> \[‘eV’, ‘kp’, ‘kx’, ‘kz’, ‘ky’, ‘phi’, ‘beta’, ‘theta\] :param arr:
> :param axis: :param smooth\_fn: :param order: Specifies how many
> derivatives to take :return:
