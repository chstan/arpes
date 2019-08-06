import xarray as xr

from arpes.provenance import update_provenance
from arpes.utilities import lift_dataarray_to_generic

__all__ = ('mean_and_deviation',)


@update_provenance('Calculate mean and standard deviation for observation axis')
@lift_dataarray_to_generic
def mean_and_deviation(data: xr.DataArray, axis=None, name=None):
    """
    Calculates the mean and standard deviation of a DataArray along an axis
    that corresponds to individual observations. This axis can be passed or
    inferred from a set of standard observation-like axes.

    New data variables are created with names `{name}` and `{name}_std`.
    If a name is not attached to the DataArray, it should be provided.

    :param data:
    :param axis:
    :param name:
    :return:
    """
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