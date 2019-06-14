import copy
import collections
import functools
import numpy as np

import xarray as xr

__all__ = ('replace_coords', 'disambiguate_coordinates',)


def disambiguate_coordinates(datasets, possibly_clashing_coordinates):
    coords_set = collections.defaultdict(list)
    for d in datasets:
        for c in possibly_clashing_coordinates:
            if c in d.coords:
                coords_set[c].append(d.coords[c])

    conflicted = []
    for c in possibly_clashing_coordinates:
        different_coords = coords_set[c]
        if len(different_coords) == 0:
            continue

        if not functools.reduce(lambda x, y: (
                    np.array_equal(x[1], y) and x[0], y),
                                different_coords, (True, different_coords[0]))[0]:
            conflicted.append(c)

    def clarify_dimensions(dims, sname):
        return [d if d not in conflicted else d + '-' + sname for d in dims]

    after_deconflict = []
    for d in datasets:
        spectrum_name = list(d.data_vars.keys())[0]
        to_rename = {name: name + '-' + spectrum_name for name in d.dims if name in conflicted}
        after_deconflict.append(d.rename(to_rename))

    return after_deconflict


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
