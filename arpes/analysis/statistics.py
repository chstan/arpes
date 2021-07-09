"""Contains utilities for performing statistical operations in spectra and DataArrays."""

import xarray as xr
from arpes.provenance import update_provenance
from arpes.utilities import lift_dataarray_to_generic

__all__ = ("mean_and_deviation",)


@update_provenance("Calculate mean and standard deviation for observation axis")
@lift_dataarray_to_generic
def mean_and_deviation(data: xr.DataArray, axis=None, name=None):
    """Calculates the mean and standard deviation of a DataArray along an axis.

    The reduced axis corresponds to individual observations of a tensor/array valued quantity.
    This axis can be passed or inferred from a set of standard observation-like axes.

    New data variables are created with names `{name}` and `{name}_std`.
    If a name is not attached to the DataArray, it should be provided.

    Args:
        data: The input data.
        axis: The name of the dimension which we should perform the reduction along.
        name: The name of the variable which should be reduced. By default, uses `data.name`.

    Returns:
        A dataset with variables corresponding to the mean and standard error of each
        relevant variable in the input DataArray.
    """
    preferred_axes = ["bootstrap", "cycle", "idx"]

    name = data.name if data.name is not None else name
    assert name is not None

    if axis is None:
        for pref_axis in preferred_axes:
            if pref_axis in data.dims:
                axis = pref_axis
                break

    assert axis in data.dims
    return xr.Dataset(
        data_vars={name: data.mean(axis), name + "_std": data.std(axis)}, attrs=data.attrs
    )
