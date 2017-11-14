import copy

import numpy as np
import xarray as xr
from scipy.ndimage import geometric_transform

from arpes.provenance import provenance
from typing import Callable

__all__ = ('flip_axis', 'normalize_dim', 'dim_normalizer', 'transform_dataarray_axis',)


def flip_axis(arr: xr.DataArray, axis_name, flip_data=True):
    coords = copy.deepcopy(arr.coords)
    coords[axis_name] = coords[axis_name][::-1]

    return xr.DataArray(
        np.flip(arr.values, arr.dims.index(axis_name)) if flip_data else arr.values,
        coords,
        arr.dims,
        attrs=arr.attrs
    )


def normalize_dim(arr: xr.DataArray, dim, keep_id=False):
    """
    Normalizes the intensity so that all values along arr.sum(dims other than ``dim``) have the same value.
    The function normalizes so that the average value of cells in the output is 1.
    :param dim_name:
    :return:
    """

    summed_arr = arr.sum([d for d in arr.dims if d != dim])
    normalized_arr = arr / (summed_arr / np.product(summed_arr.shape))

    to_return = xr.DataArray(
        normalized_arr.values,
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if not keep_id and 'id' in to_return.attrs:
        del to_return.attrs['id']

    provenance(to_return, arr, {
        'what': 'Normalize axis',
        'by': 'normalize_dim',
        'dim': dim,
    })

    return to_return


def dim_normalizer(dim_name):
    def normalize(arr: xr.DataArray):
        if dim_name not in arr.dims:
            return arr
        return normalize_dim(arr, dim_name)

    return normalize


def transform_dataarray_axis(f: Callable[[float], float], old_axis_name: str,
                             new_axis_name: str,
                             new_axis, dataset: xr.DataArray, name):
    dataset.coords[new_axis_name] = new_axis

    old_axis = dataset.raw.dims.index(old_axis_name)

    shape = list(dataset.raw.sizes.values())
    shape[old_axis] = len(new_axis)

    new_dims = list(dataset.raw.dims)
    new_dims[old_axis] = new_axis_name

    output = geometric_transform(dataset.raw.values, f, output_shape=shape, output='f', order=1)

    new_coords = dict(dataset.coords)
    new_coords.pop(old_axis_name)

    new_dataarray = xr.DataArray(output, coords=new_coords, dims=new_dims, attrs=dataset.attrs,).rename(name)
    del new_dataarray.attrs['id']

    provenance(new_dataarray, dataset, {
        'what': 'Transformed a DataArray coordinate axis',
        'by': 'transform_dataarray_axis',
        'old_axis': old_axis_name,
        'new_axis': new_axis_name,
    })
    return xr.merge([
        dataset,
        xr.DataArray(
            output,
            coords=new_coords,
            dims=new_dims,
            attrs=dataset.attrs,
        ).rename(name)
    ])