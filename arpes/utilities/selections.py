"""
This package contains utilities related to taking more complicated shaped selections around data.

Currently it houses just utilities for forming disk and annular selections out of data.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.xarray import unwrap_xarray_dict

__all__ = ('select_disk', 'select_disk_mask', 'unravel_from_mask', 'ravel_from_mask')


def ravel_from_mask(data, mask):
    """
    Selects out the data from a ND array whose points are marked true in `mask`. See also `unravel_from_mask`
    below which allows you to write back into data after you have transformed the 1D output in some way.

    These two functions are especially useful for hierarchical curve fitting where you want to rerun a fit over a subset
    of the data with a different model, such as when you know some of the data is best described by two bands rather than
    one.
    :param data:
    :param mask:
    :return:
    """
    return data.stack(stacked=mask.dims).where(mask.stack(stacked=mask.dims), drop=True)


def unravel_from_mask(template, mask, values, default=np.nan):
    """
    Creates an array
    :param template:
    :param mask:
    :param values:
    :param default:
    :return:
    """
    dest = template * 0 + 1
    dest_mask = np.logical_not(np.isnan(template.stack(stacked=template.dims).where(mask.stack(stacked=template.dims)).values))
    dest = (dest * default).stack(stacked=template.dims)
    dest.values[dest_mask] = values
    return dest.unstack('stacked')


def _normalize_point(data, around, **kwargs):
    collected_kwargs = {k: kwargs[k] for k in data.dims if k in kwargs}

    if around:
        if isinstance(around, xr.Dataset):
            around = unwrap_xarray_dict({d: around[d] for d in data.dims})
    else:
        around = collected_kwargs

    assert set(around.keys()) == set(data.dims)
    return around


def select_disk_mask(data: DataType, radius, outer_radius=None, around: Optional[Union[Dict, xr.Dataset]] = None,
                     flat=False, **kwargs) -> np.ndarray:
    """
    A complement to select_disk which only generates the mask.

    Selects the data in a disk around the point described by `around` and `kwargs`. A point is a labelled
    collection of coordinates that matches all of the dimensions of `data`. The coordinates can either be
    passed through a dict as `around`, as the coordinates of a Dataset through `around` or explicitly in
    keyword argument syntax through `kwargs`. The radius for the disk is specified through the required
    `radius` parameter.

    Returns the ND mask that represents the filtered coordinates.
    :param data:
    :param around:
    :param flat: Whether to return the mask as a 1D (raveled) mask (flat=True) or as a ND mask with the
                 same dimensionality as the input data (flat=False).
    :param kwargs:
    :return:
    """
    if outer_radius is not None and radius > outer_radius:
        radius, outer_radius = outer_radius, radius

    data = normalize_to_spectrum(data)
    around = _normalize_point(data, around, **kwargs)

    raveled = data.T.ravel()

    dim_order = list(around.keys())
    dist = np.sqrt(np.sum(np.stack([(raveled[d] - around[d]) ** 2 for d in dim_order], axis=1), axis=1))

    mask = dist <= radius
    if outer_radius is not None:
        mask = np.logical_or(mask, dist > outer_radius)

    if flat:
        return mask

    return mask.reshape(data.shape[::-1])


def select_disk(data: DataType, radius, outer_radius=None, around: Optional[Union[Dict, xr.Dataset]] = None,
                invert=False, **kwargs) \
        -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Selects the data in a disk (or annulus if `outer_radius` is provided) around the point described
    by `around` and `kwargs`. A point is a labeled collection of coordinates that matches all of the dimensions
    of `data`. The coordinates can either be passed through a dict as `around`, as the coordinates of a Dataset through
    `around` or explicitly in keyword argument syntax through `kwargs`. The radius for the disk is specified through
    the required `radius` parameter.

    Data is returned as a tuple with the type Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]
    containing a dictionary with the filtered lists of coordinates, an array with the original data
    values at these coordinates, and finally an array of the distances to the requested point.
    :param data:
    :param radius:
    :param outer_radius:
    :param around:
    :param invert:
    :param kwargs:
    :return:
    """

    data = normalize_to_spectrum(data)
    around = _normalize_point(data, around, **kwargs)
    mask = select_disk_mask(data, radius, outer_radius=outer_radius, around=around, flat=True)

    if invert:
        mask = np.logical_not(mask)

    # at this point, around is now a dictionary specifying a point to do the selection around
    raveled = data.T.ravel()
    data_arr = raveled['data']

    dim_order = list(around.keys())
    dist = np.sqrt(np.sum(np.stack([(raveled[d] - around[d]) ** 2 for d in dim_order], axis=1), axis=1))

    masked_coords = {d: cs[mask] for d, cs in raveled.items()}
    return masked_coords, masked_coords['data'], dist[mask]

