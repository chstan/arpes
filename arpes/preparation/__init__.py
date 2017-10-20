import copy
import math
from typing import Callable

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from scipy.ndimage import geometric_transform
from skimage import feature

from arpes.provenance import provenance
from arpes.utilities import get_spectrometer


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

def build_KE_coords_to_time_pixel_coords(dataset: xr.Dataset, interpolation_axis):
    t_pixel_axis = dataset.raw.dims.index('t_pixels')

    conv = (0.5) * (9.11e6) * 0.5 * (get_spectrometer(dataset)['length'] ** 2) / 1.6
    time_res = 0.17 # this is only approximate

    def KE_coords_to_time_pixel_coords(coords):
        """
        Like ``KE_coords_to_time_coords`` but it converts to the raw timing pixels off of
        a DLD instead to the unitful values that we receive from the Spin-ToF DAQ
        :param coords: tuple of coordinates
        :return: new tuple of converted coordinates
        """
        kinetic_energy_pixel = coords[t_pixel_axis]
        kinetic_energy = interpolation_axis[kinetic_energy_pixel]
        real_timing = math.sqrt(conv / kinetic_energy)
        pixel_timing = (real_timing - dataset.attrs['timing_offset']) / time_res
        coords_list = list(coords)
        coords_list[t_pixel_axis] = pixel_timing

        return tuple(coords_list)
    return KE_coords_to_time_pixel_coords


def build_KE_coords_to_time_coords(dataset: xr.Dataset):
    t_axis = dataset.raw.dims.index('t')

    conv = (0.5) * (9.11e6) * 0.5 * (get_spectrometer(dataset)['length'] ** 2) / 1.6

    def KE_coords_to_time_coords(coords):
        """
        Used to convert the timing coordinates off of the spin-ToF to kinetic energy coordinates.
        As the name suggests, because of how scipy.ndimage interpolates, we require the inverse
        coordinate transform.

        All coordinates except for the energy coordinate are left untouched.
        :param coords: tuple of coordinates
        :return: new tuple of converted coordinates
        """
        kinetic_energy = coords[t_axis]
        real_timing = math.sqrt(conv / kinetic_energy)
        coords_list = list(coords)
        coords_list[t_axis] = real_timing

        return coords

    return KE_coords_to_time_coords

def process_DLD(dataset: xr.Dataset):
    e_min = 1
    ke_axis = np.linspace(e_min, dataset.attrs['E_max'], (dataset.attrs['E_max'] - e_min) / dataset.attrs['dE'])
    dataset = transform_dataarray_axis(
        build_KE_coords_to_time_pixel_coords(dataset, ke_axis),
        't_pixels', 'KE', ke_axis, dataset, 'KE_spectrum'
    )

    return dataset


def replace_coords(arr: xr.DataArray, new_coords, mapping):
    coords = dict(copy.deepcopy(arr.coords))
    dims = list(copy.deepcopy(arr.dims))
    for old_dim, new_dim in mapping:
        coords[new_dim] = new_coords[new_dim]
        del coords[old_dim]
        dims[dims.index(old_dim)] = new_dim

    return xr.DataArray(
        arr.values,
        coords,
        dims,
        attrs=arr.attrs,
    )


def normalize_dim(arr: xr.DataArray, dim, keep_id=False):
    normalized_arr = arr / arr.sum([d for d in arr.dims if d != dim])

    to_return = xr.DataArray(
        normalized_arr.values,
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if not keep_id:
        del to_return.attrs['id']

    provenance(to_return, arr, {
        'what': 'Normalize axis',
        'by': 'normalize_dim',
        'dim': dim,
    })

    return to_return


def flip_axis(arr: xr.DataArray, axis_name, flip_data=True):
    import pdb
    pdb.set_trace()
    coords = copy.deepcopy(arr.coords)
    coords[axis_name] = coords[axis_name][::-1]

    return xr.DataArray(
        np.flip(arr.values, arr.dims.index(axis_name)) if flip_data else arr.values,
        coords,
        arr.dims,
        attrs=arr.attrs
    )


def infer_center_pixel(arr: xr.DataArray):
    near_ef = arr.sel(eV=slice(-0.1, 0)).sum([d for d in arr.dims if d not in ['pixel']])
    embed_size = 20
    embedded = np.ndarray(shape=[embed_size] + list(near_ef.values.shape))
    embedded[:] = near_ef.values
    embedded = ndi.gaussian_filter(embedded, embed_size / 3)

    edges = feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True,
                          low_threshold=0.2) * 1
    edges = np.where(edges[int(embed_size / 2)] == 1)

    return float((np.max(edges) + np.min(edges)) / 2 + np.min(arr.coords['pixel']))


def dim_normalizer(dim_name):
    def normalize(arr: xr.DataArray):
        if dim_name not in arr.dims:
            return arr
        return normalize_dim(arr, dim_name)

    return normalize