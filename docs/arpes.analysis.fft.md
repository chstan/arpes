# arpes.analysis.fft module

This module contains monkey-patched versions of functions from xrft
until improvements are made upstream. We don’t generally use this module
too much anyway, and it is not a default import.

**arpes.analysis.fft.fft\_filter(data: xarray.core.dataarray.DataArray,
stops)**

> Applies a brick wall filter at region in `stops` in the Fourier
> transform of data. Use with care. :param data: :param stops: :return:
