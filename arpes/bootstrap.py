import numpy as np
import random
import copy
import xarray as xr
import functools

from arpes.analysis.sarpes import to_intensity_polarization
from typing import Union
from tqdm import tqdm_notebook
from arpes.utilities.region import normalize_region
from arpes.utilities.normalize import normalize_to_spectrum
from arpes.typing import DataType
from arpes.utilities import lift_dataarray_to_generic

__all__ = ('bootstrap', 'estimate_prior_adjustment',
           'resample_true_counts', 'bootstrap_counts',
           'bootstrap_intensity_polarization',)


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


@lift_dataarray_to_generic
def resample_cycle(data: xr.DataArray, **kwargs):
    """
    Perform a non-parametric bootstrap using a cycle coordinate for statistically independent observations.
    :param data:
    :param kwargs:
    :return:
    """

    n_cycles = len(data.cycle)
    which = [random.randint(0, n_cycles - 1) for _ in range(n_cycles)]

    resampled = data.isel(cycle=which).sum('cycle', keep_attrs=True)

    if 'id' in resampled.attrs:
        del resampled.attrs['id']

    return resampled

@lift_dataarray_to_generic
def resample(data: xr.DataArray, prior_adjustment=1, **kwargs):
    resampled = xr.DataArray(
        np.random.poisson(lam=data.values * prior_adjustment, size=data.values.shape),
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs
    )

    if 'id' in resampled.attrs:
        del resampled.attrs['id']

    return resampled


@lift_dataarray_to_generic
def resample_true_counts(data: xr.DataArray) -> xr.DataArray:
    """
    Resamples histogrammed data where each count represents an actual electron.
    :param data:
    :return:
    """

    resampled = xr.DataArray(
        np.random.poisson(lam=data.values, size=data.values.shape),
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs,
    )

    if 'id' in resampled.attrs:
        del resampled.attrs['id']

    return resampled

@lift_dataarray_to_generic
def bootstrap_counts(data: DataType, N=1000, name=None) -> xr.Dataset:
    """
    Parametric bootstrap for the number of counts in each detector channel for a
    time of flight/DLD detector, where each count represents an actual particle.

    This function also introspects the data passed to determine whether there is a
    spin degree of freedom, and will bootstrap appropriately.

    Currently we build all the samples at once instead of using a rolling algorithm.
    :param data:
    :return:
    """

    assert (data.name is not None or name is not None)
    name = data.name if data.name is not None else name

    desc_fragment = ' {}'.format(name)

    resampled_sets = []
    for _ in tqdm_notebook(range(N), desc='Resampling{}...'.format(desc_fragment)):
        resampled_sets.append(resample_true_counts(data))

    resampled_arr = np.stack([s.values for s in resampled_sets], axis=0)
    std = np.std(resampled_arr, axis=0)
    std = xr.DataArray(std, data.coords, data.dims)
    mean = np.mean(resampled_arr, axis=0)
    mean = xr.DataArray(mean, data.coords, data.dims)

    vars = {}
    vars[name] = mean
    vars[name + '_std'] = std

    return xr.Dataset(data_vars=vars, coords=data.coords, attrs=data.attrs.copy())


def bootstrap_intensity_polarization(data, N=100):
    """
    Uses the parametric bootstrap to get uncertainties on the intensity and polarization of ToF-SARPES data.
    :param data:
    :return:
    """

    bootstrapped_polarization = bootstrap(to_intensity_polarization)
    return bootstrapped_polarization(data, N=N)


def bootstrap(fn, skip=None, resample_method=None):
    if skip is None:
        skip = []

    skip = set(skip)

    if resample_method is None:
        resample_fn = resample
    elif resample_method == 'cycle':
        resample_fn = resample_cycle

    def bootstrapped(*args, N=20, prior_adjustment=1, **kwargs):
        # examine args to determine which to resample
        resample_indices = [i for i, arg in enumerate(args) if isinstance(arg, (xr.DataArray, xr.Dataset)) and i not in skip]
        data_is_arraylike = False

        runs = []

        def get_label(i):
            if isinstance(args[i], xr.Dataset):
                return 'xr.Dataset: [{}]'.format(', '.join(args[i].data_vars.keys()))
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
                new_args[i] = resample_fn(args[i], prior_adjustment=prior_adjustment)
            for k in resample_kwargs:
                new_kwargs[k] = resample_fn(kwargs[k], prior_adjustment=prior_adjustment)


            run = fn(*new_args, **new_kwargs)
            if isinstance(run, (xr.DataArray, xr.Dataset)):
                data_is_arraylike = True
            runs.append(run)

        if data_is_arraylike:
            for i, run in enumerate(runs):
                run.coords['bootstrap'] = i

            return xr.concat(runs, dim='bootstrap')

        return runs

    return functools.wraps(fn)(bootstrapped)