import numpy as np
import copy
import xarray as xr
import functools
from typing import Union
from tqdm import tqdm_notebook
from arpes.utilities.region import normalize_region
from arpes.utilities.normalize import normalize_to_spectrum
from arpes.typing import DataType

__all__ = ('bootstrap', 'estimate_prior_adjustment',)


def estimate_prior_adjustment(data: DataType, region: Union[dict, str]=None):
    """
    Estimates the parameters of a distribution generating the intensity
    histogram of pixels in a spectrum. In a perfectly linear, single-electron
    single-count detector, this would be a poisson distribution with
    \lambda=mean(counts) over the window. Despite this, we can estimate \lambda
    phenomenologically and verify that a Poisson distribution provides a good
    prior for the data, allowing us to perform statistical bootstrapping.

    You should use this with a spectrum that has uniform intensity, i.e. with a
    copper reference or similar.

    :param data:
    :return: returns sigma / mu, adjustment factor for the Poisson distribution
    """
    data = normalize_to_spectrum(data)

    if region is None:
        region = 'copper_prior'

    region = normalize_region(region)

    if 'cycle' in data.dims:
        data = data.sum('cycle')

    data = data.S.zero_spectrometer_edges().S.region_sel(region)
    values = data.values.ravel()
    values = values[np.where(values)]
    return np.std(values) / np.mean(values)


def resample(data: xr.DataArray, prior_adjustment=1):
    resampled = xr.DataArray(
        np.random.poisson(lam=data.values * prior_adjustment, size=data.values.shape),
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs
    )

    if 'id' in resampled.attrs:
        del resampled.attrs['id']

    return resampled

def bootstrap(fn, skip=None):
    if skip is None:
        skip = []

    skip = set(skip)

    def bootstrapped(*args, N=20, prior_adjustment=1, **kwargs):
        # examine args to determine which to resample
        resample_indices = [i for i, arg in enumerate(args) if isinstance(arg, xr.DataArray) and i not in skip]
        data_is_arraylike = False

        runs = []

        def get_label(i):
            if args[i].name:
                return args[i].name

            try:
                return args[i].attrs['id']
            except KeyError:
                return 'Label-less DataArray'

        print('Resampling args: {}'.format(','.join([get_label(i) for i in resample_indices])))

        # examine kwargs to determine which to resample
        resample_kwargs = [k for k, v in kwargs.items() if isinstance(v, xr.DataArray) and k not in skip]
        print('Resampling kwargs: {}'.format(','.join(resample_kwargs)))

        print('Fair warning 1: Make sure you understand whether it is appropriate to resample your data.')
        print('Fair warning 2: Ensure that the data to resample is in a DataArray and not a Dataset')

        for _ in tqdm_notebook(range(N), desc='Resampling...'):
            new_args = list(args)
            new_kwargs = copy.copy(kwargs)
            for i in resample_indices:
                new_args[i] = resample(args[i], prior_adjustment=prior_adjustment)
            for k in resample_kwargs:
                new_kwargs[k] = resample(kwargs[k], prior_adjustment=prior_adjustment)


            run = fn(*new_args, **new_kwargs)
            if isinstance(run, xr.DataArray):
                data_is_arraylike = True
            runs.append(run)

        if data_is_arraylike:
            coords = dict(runs[0].coords)
            coords['bootstrap'] = np.array(range(N))
            dims = runs[0].dims
            dims = tuple(['bootstrap'] + list(dims))
            attrs = runs[0].attrs
            original_shape = runs[0].values.shape
            return xr.DataArray(
                np.concatenate(runs).reshape(tuple([N] + list(original_shape))),
                coords=coords,
                dims=dims,
                attrs=attrs
            )
        return runs

    return functools.wraps(fn)(bootstrapped)