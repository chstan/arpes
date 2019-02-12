import copy
import functools

import numpy as np
import xarray as xr

from arpes.typing import DataType

from arpes.utilities.normalize import normalize_to_spectrum
from scipy.ndimage import geometric_transform

from arpes.provenance import provenance
from arpes.utilities import lift_dataarray_to_generic

__all__ = ('flip_axis', 'normalize_dim', 'dim_normalizer', 'transform_dataarray_axis',
           'normalize_total', 'sort_axis',)


def sort_axis(data: xr.DataArray, axis_name):
    assert(isinstance(data, xr.DataArray))
    copied = data.copy(deep=True)
    coord = data.coords[axis_name].values
    order = np.argsort(coord)
    print(order)
    copied.values = np.take(copied.values, order, axis=list(data.dims).index(axis_name))
    copied.coords[axis_name] = np.sort(copied.coords[axis_name])
    return copied


def flip_axis(arr: xr.DataArray, axis_name, flip_data=True):
    coords = copy.deepcopy(arr.coords)
    coords[axis_name] = coords[axis_name][::-1]

    return xr.DataArray(
        np.flip(arr.values, arr.dims.index(axis_name)) if flip_data else arr.values,
        coords,
        arr.dims,
        attrs=arr.attrs
    )

def soft_normalize_dim(arr: xr.DataArray, dim_or_dims, keep_id=False, amp_limit=100):
    dims = dim_or_dims
    if isinstance(dim_or_dims, str):
        dims = [dims]

    summed_arr = arr.fillna(arr.mean()).sum([d for d in arr.dims if d not in dims])
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
        'what': 'Normalize axis or axes',
        'by': 'normalize_dim',
        'dims': dims,
    })

    return to_return

@lift_dataarray_to_generic
def normalize_dim(arr: DataType, dim_or_dims, keep_id=False):
    """
    Normalizes the intensity so that all values along arr.sum(dims other than those in ``dim_or_dims``)
    have the same value. The function normalizes so that the average value of cells in
    the output is 1.
    :param dim_name:
    :return:
    """

    dims = dim_or_dims
    if isinstance(dim_or_dims, str):
        dims = [dims]

    summed_arr = arr.fillna(arr.mean()).sum([d for d in arr.dims if d not in dims])
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
        'what': 'Normalize axis or axes',
        'by': 'normalize_dim',
        'dims': dims,
    })

    return to_return


def normalize_total(data: DataType):
    data = normalize_to_spectrum(data)

    return data / (data.sum(data.dims) / 1000000)


def dim_normalizer(dim_name):
    def normalize(arr: xr.DataArray):
        if dim_name not in arr.dims:
            return arr
        return normalize_dim(arr, dim_name)

    return normalize


def transform_dataarray_axis(f, old_axis_name: str,
                             new_axis_name: str, new_axis,
                             dataset: xr.DataArray, prep_name,
                             transform_spectra=None, remove_old=True):

    ds = dataset.copy()
    if transform_spectra is None:
        # transform *all* DataArrays in the dataset that have old_axis_name in their dimensions
        transform_spectra = {k: v for k, v in ds.data_vars.items()
                             if old_axis_name in v.dims}

    ds.coords[new_axis_name] = new_axis

    new_dataarrays = []
    for name in transform_spectra.keys():
        dr = ds[name]

        old_axis = dr.dims.index(old_axis_name)
        shape = list(dr.sizes.values())
        shape[old_axis] = len(new_axis)
        new_dims = list(dr.dims)
        new_dims[old_axis] = new_axis_name

        g = functools.partial(f, axis=old_axis)
        output = geometric_transform(dr.values, g, output_shape=shape, output='f', order=1)

        new_coords = dict(dr.coords)
        new_coords.pop(old_axis_name)

        new_dataarray = xr.DataArray(output, coords=new_coords, dims=new_dims,
                                     attrs=dr.attrs.copy(), name=prep_name(dr.name))
        new_dataarrays.append(new_dataarray)
        if 'id' in new_dataarray.attrs:
            del new_dataarray.attrs['id']

        if remove_old:
            del ds[name]
        else:
            assert(prep_name(name) != name and "You must make sure names don't collide")

    new_ds = xr.merge([
        ds,
        *new_dataarrays
    ])

    new_ds.attrs.update(ds.attrs.copy())

    if 'id' in new_ds:
        del new_ds.attrs['id']

    provenance(new_ds, dataset, {
        'what': 'Transformed a Dataset coordinate axis',
        'by': 'transform_dataarray_axis',
        'old_axis': old_axis_name,
        'new_axis': new_axis_name,
        'transformed_vars': list(transform_spectra.keys()),
    })

    return new_ds
