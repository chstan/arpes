import xarray as xr

from arpes.utilities import lift_dataarray_to_generic

__all__ = ('mean_and_deviation',)


@lift_dataarray_to_generic
def mean_and_deviation(data: xr.DataArray, axis=None, name=None):
    preferred_axes = ['bootstrap', 'cycle', 'idx']

    name = data.name if data.name is not None else name
    assert(name is not None)

    if axis is None:
        for pref_axis in preferred_axes:
            if pref_axis in data.dims:
                axis = pref_axis
                break

    assert(axis in data.dims)
    return xr.Dataset(data_vars={name: data.mean(axis), name + '_std': data.std(axis)}, attrs=data.attrs)