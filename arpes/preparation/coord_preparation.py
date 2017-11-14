import copy

import xarray as xr

__all__ = ('replace_coords',)

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
