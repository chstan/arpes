import xarray as xr

__all__ = ['convert_to_kspace_forward']


def convert_coordinates_forward(arr: xr.DataArray, dimension_to_convert):
    target_coordinates = dict(arr.coords)

    if dimension_to_convert == 'phi':
        return {}
    if dimension_to_convert == 'polar':
        return {}
    if dimension_to_convert == 'hv':
        return {}


def convert_to_kspace_forward(arr: xr.DataArray, **kwargs):
    target_dims = list(arr.dims)

    dim_conversion = {
        'phi': 'kp',
        'polar': 'kp',
        'hv': 'kz',
    }
    convertable = set(dim_conversion.keys())
    to_convert = list(convertable.intersection(set(arr.dims)))
    assert (len(to_convert) == 1)

    target_dims = [dim_conversion.get(d, d) for d in target_dims]

    # can also do fancy shit like convert attributes that have units, but this is not
    # necessary for the moment

    converted = xr.DataArray(
        arr.values.copy(),
        convert_coordinates_forward(arr, to_convert[0]),
        target_dims,
        attrs=arr.attrs,
    )
    if 'id' in converted.attrs:
        del converted.attrs['id']

    return converted
