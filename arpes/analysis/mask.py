from arpes.typing import DataType
from matplotlib.path import Path
import xarray as xr

import numpy as np

from arpes.utilities import normalize_to_spectrum
from arpes.provenance import update_provenance

__all__ = ('polys_to_mask', 'apply_mask', 'raw_poly_to_mask',
           'apply_mask_to_coords',)


def raw_poly_to_mask(poly):
    """
    There's not currently much metadata attached to masks, but this is
    around if we ever decide that we need to implement more
    complicated masking schemes.

    In particular, we might want to store also whether the interior
    or exterior is the masked region, but this is functionally achieved
    for now with the `invert` flag in other functions.

    :param poly: Polygon implementing a masked region.
    :return:
    """
    return {
        'poly': poly,
    }


def polys_to_mask(mask_dict, coords, shape, radius=None, invert=False):
    """
    Converts a mask definition in terms of the underlying polygon to a True/False
    mask array using the coordinates and shape of the target data.

    This process "specializes" a mask to a particular shape, whereas masks given by
    polygon definitions are general to any data with appropriate dimensions, because
    waypoints are given in unitful values rather than index values.
    :param mask_dict:
    :param coords:
    :param shape:
    :param radius:
    :param invert:
    :return:
    """

    dims = mask_dict['dims']
    polys = mask_dict['polys']

    polys = [[[np.searchsorted(coords[dims[i]], coord) for i, coord in enumerate(p)] for p in poly] for poly in polys]

    mask_grids = np.meshgrid(*[np.arange(s) for s in shape])
    mask_grids = tuple(k.flatten() for k in mask_grids)

    points = np.vstack(mask_grids).T

    mask = None
    for poly in polys:
        grid = Path(poly).contains_points(points, radius=radius or 0)
        grid = grid.reshape(list(shape)[::-1]).T

        if mask is None:
            mask = grid
        else:
            mask = np.logical_or(mask, grid)

    if invert:
        mask = np.logical_not(mask)

    return mask


def apply_mask_to_coords(data: xr.Dataset, mask, dims, invert=True):
    p = Path(mask['poly'])

    as_array = np.stack([data.data_vars[d].values for d in dims], axis=-1)
    shape = as_array.shape
    dest_shape = shape[:-1]
    new_shape = [np.prod(dest_shape), len(dims)]

    mask = p.contains_points(as_array.reshape(new_shape)).reshape(dest_shape)
    if invert:
        mask = np.logical_not(mask)

    return mask


@update_provenance('Apply boolean mask to data')
def apply_mask(data: DataType, mask, replace=np.nan, radius=None, invert=False):
    """
    Applies a logical mask, i.e. one given in terms of polygons, to a specific
    piece of data. This can be used to set values outside or inside a series of
    polygon masks to a given value or to NaN.

    Expanding or contracting the masked region can be accomplished with the
    radius argument, but by default strict inclusion is used.

    Some masks include a `fermi` parameter which allows for clipping the detector
    boundaries in a semi-automated fashion. If this is included, only 200meV above the Fermi
    level will be included in the returned data. This helps to prevent very large
    and undesirable regions filled with only the replacement value which can complicate
    automated analyses that rely on masking.

    :param data: Data to mask.
    :param mask: Logical definition of the mask, appropriate for passing to `polys_to_mask`
    :param replace: The value to substitute for pixels masked.
    :param radius: Radius by which to expand the masked area.
    :param invert: Allows logical inversion of the masked parts of the data. By default,
                   the area inside the polygon sequence is replaced by `replace`.
    :return:
    """
    data = normalize_to_spectrum(data)
    fermi = mask.get('fermi')

    if isinstance(mask, dict):
        dims = mask.get('dims', data.dims)
        mask = polys_to_mask(mask, data.coords, [s for i, s in enumerate(data.shape) if data.dims[i] in dims], radius=radius, invert=invert)

    masked_data = data.copy(deep=True)
    masked_data.values = masked_data.values * 1.0
    masked_data.values[mask] = replace

    if fermi is not None:
        return masked_data.sel(eV=slice(None, fermi + 0.2))

    return masked_data
