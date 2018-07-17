import xarray as xr

from arpes.typing import DataType

__all__ = ('normalize_to_spectrum',)

def normalize_to_spectrum(data: DataType):
    if isinstance(data, xr.Dataset):
        if 'up' in data.data_vars:
            return data.up

        return data.S.spectrum

    # not guaranteed to be a spectrum, but close enough
    return data
