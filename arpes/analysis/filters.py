import copy

import numpy as np
from scipy import ndimage

import xarray as xr
from arpes.provenance import provenance

__all__ = ('gaussian_filter_arr', 'gaussian_filter', 'boxcar_filter_arr', 'boxcar_filter',)


def gaussian_filter_arr(arr: xr.DataArray, sigma=None, n=1, default_size=1):
    """
    Functionally wraps scipy.ndimage.filters.gaussian_filter with the advantage that the sigma
    is coordinate aware.

    :param arr:
    :param sigma: Kernel sigma, specified in terms of axis units. An axis that is not specified
                  will have a kernel width of `default_size` in index units.
    :param n: Repeats n times.
    :param default_size: Changes the default kernel width for axes not specified in `sigma`. Changing this
                         parameter and leaving `sigma` as None allows you to smooth with an even-width
                         kernel in index-coordinates.
    :return: xr.DataArray: smoothed data.
    """
    if sigma is None:
        sigma = {}

    sigma = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in sigma.items()}
    for dim in arr.dims:
        if dim not in sigma:
            sigma[dim] = default_size

    sigma = tuple(sigma[k] for k in arr.dims)

    values = arr.values
    for _ in range(n):
        values = ndimage.filters.gaussian_filter(values, sigma)

    filtered_arr = xr.DataArray(
        values,
        arr.coords,
        arr.dims,
        attrs=copy.deepcopy(arr.attrs)
    )

    if 'id' in filtered_arr.attrs:
        del filtered_arr.attrs['id']

        provenance(filtered_arr, arr, {
            'what': 'Gaussian filtered data',
            'by': 'gaussian_filter_arr',
            'sigma': sigma,
        })

    return filtered_arr


def gaussian_filter(sigma=None, n=1):
    """
    A partial application of `gaussian_filter_arr` that can be passed to derivative analysis functions.
    :param sigma:
    :param n:
    :return:
    """

    def f(arr):
        return gaussian_filter_arr(arr, sigma, n)

    return f


def boxcar_filter(size=None, n=1):
    """
    A partial application of `boxcar_filter_arr` that can be passed to derivative analysis functions.
    :param size:
    :param n:
    :return:
    """

    def f(arr):
        return boxcar_filter_arr(arr, size, n)

    return f


def boxcar_filter_arr(arr: xr.DataArray, size=None, n=1, default_size=1, skip_nan=True):
    """
    Functionally wraps scipy.ndimage.filters.gaussian_filter with the advantage that the sigma
    is coordinate aware.

    :param arr:
    :param size: Kernel size, specified in terms of axis units. An axis that is not specified
                 will have a kernel width of `default_size` in index units.
    :param n: Repeats n times.
    :param default_size: Changes the default kernel width for axes not specified in `sigma`. Changing this
                         parameter and leaving `sigma` as None allows you to smooth with an even-width
                         kernel in index-coordinates.
    :param skip_nan: By default, masks parts of the data which are NaN to prevent poor filter results.
    :return: xr.DataArray: smoothed data.
    """

    if size is None:
        size = {}

    size = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in size.items()}
    for dim in arr.dims:
        if dim not in size:
            size[dim] = default_size

    size = tuple(size[k] for k in arr.dims)

    if skip_nan:
        nan_mask = np.copy(arr.values) * 0 + 1
        nan_mask[arr.values != arr.values] = 0
        filtered_mask = ndimage.filters.uniform_filter(nan_mask, size)

        values = np.copy(arr.values)
        values[values != values] = 0

        for _ in range(n):
            values = ndimage.filters.uniform_filter(values, size) / filtered_mask
            values[nan_mask == 0] = 0
    else:
        for i in range(n):
            values = ndimage.filters.uniform_filter(values, size)

    filtered_arr = xr.DataArray(
        values,
        arr.coords,
        arr.dims,
        attrs=copy.deepcopy(arr.attrs)
    )

    if 'id' in arr.attrs:
        del filtered_arr.attrs['id']

        provenance(filtered_arr, arr, {
            'what': 'Boxcar filtered data',
            'by': 'boxcar_filter_arr',
            'size': size,
            'skip_nan': skip_nan,
        })

    return filtered_arr
