import copy

import numpy as np
import xarray as xr
from scipy import ndimage

from arpes.provenance import provenance

__all__ = ('gaussian_filter_arr', 'gaussian_filter', 'boxcar_filter_arr', 'boxcar_filter',)


def gaussian_filter_arr(arr: xr.DataArray, sigma=None, n=1, default_size=1):
    if sigma is None:
        sigma = {}

    sigma = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in sigma.items()}
    for dim in arr.dims:
        if dim not in sigma:
            sigma[dim] = default_size

    sigma = tuple(sigma[k] for k in arr.dims)

    values = arr.values
    for i in range(n):
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
    def f(arr):
        return gaussian_filter_arr(arr, sigma, n)

    return f


def boxcar_filter(size=None, n=1):
    def f(arr):
        return boxcar_filter_arr(arr, size, n)

    return f

def boxcar_filter_arr(arr: xr.DataArray, size=None, n=1, default_size=1, skip_nan=True):
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

        for i in range(n):
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
