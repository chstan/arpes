from arpes.typing import DataType
from matplotlib.path import Path

import numpy as np

from utilities import normalize_to_spectrum

__all__ = ('polys_to_mask', 'apply_mask',)


def polys_to_mask(mask_dict, coords, shape, radius=None, invert=False):
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


def apply_mask(data: DataType, mask, replace=np.nan, radius=None, invert=False):
    data = normalize_to_spectrum(data)
    fermi = mask.get('fermi')

    if isinstance(mask, dict):
        dims = mask['dims']
        mask = polys_to_mask(mask, data.coords, [s for i, s in enumerate(data.shape) if data.dims[i] in dims], radius=radius, invert=invert)

    masked_data = data.copy(deep=True)
    masked_data.values = masked_data.values * 1.0
    masked_data.values[mask] = replace

    if fermi is not None:
        return masked_data.sel(eV=slice(None, fermi + 0.2))

    return masked_data
