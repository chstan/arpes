from typing import Dict, Any

import xarray as xr

from arpes.typing import DataType

__all__ = ('apply_dataarray', 'lift_datavar_attrs', 'lift_dataarray_attrs', 'lift_dataarray',
           'unwrap_xarray_item', 'unwrap_xarray_dict')


def unwrap_xarray_item(item):
    """
    Unwraps something that might or might not be an xarray like with .item()
    attribute.

    This is especially helpful for dealing with unwrapping coordinates which might
    be floating point-like or might be array-like.

    :param item:
    :return:
    """
    try:
        return item.item()
    except (AttributeError, ValueError):
        return item


def unwrap_xarray_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Useful for unwrapping coordinate dicts where the values might be a bare type:
    like a float or an int, but might also be a wrapped array-like for instance
    xr.DataArray. Even worse, we might have wrapped bare values!

    :param d:
    :return:
    """
    return {k: unwrap_xarray_item(v) for k, v in d.items()}


def apply_dataarray(arr: xr.DataArray, f, *args, **kwargs):
    return xr.DataArray(
        f(arr.values, *args, **kwargs),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )


def lift_dataarray(f):
    """
    Lifts a function that operates on an np.ndarray's values to one that
    acts on the values of an xr.DataArray
    :param f:
    :return: g: Function operating on an xr.DataArray
    """

    def g(arr: xr.DataArray, *args, **kwargs):
        return apply_dataarray(arr, f, *args, **kwargs)

    return g


def lift_dataarray_attrs(f):
    """
    Lifts a function that operates on a dictionary to a function that acts on the
    attributes of an xr.DataArray, producing a new xr.DataArray. Another option
    if you don't need to create a new DataArray is to modify the attributes.
    :param f:
    :return: g: Function operating on the attributes of an xr.DataArray
    """

    def g(arr: xr.DataArray, *args, **kwargs):
        return xr.DataArray(
            arr.values,
            arr.coords,
            arr.dims,
            attrs=f(arr.attrs, *args, **kwargs)
        )

    return g


def lift_datavar_attrs(f):
    """
    Lifts a function that operates on a dictionary to a function that acts on the
    attributes of all the datavars in a xr.Dataset, as well as the Dataset attrs
    themselves.
    :param f: Function to apply
    :return:
    """

    def g(data: DataType, *args, **kwargs):
        arr_lifted = lift_dataarray_attrs(f)
        if isinstance(data, xr.DataArray):
            return arr_lifted(data, *args, **kwargs)

        var_names = list(data.data_vars.keys())
        new_vars = {k: arr_lifted(data[k], *args, **kwargs) for k in var_names}
        new_root_attrs = f(data.attrs, *args, **kwargs)

        return xr.Dataset(new_vars, data.coords, new_root_attrs)

    return g
