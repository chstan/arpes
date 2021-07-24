"""Utilities to programmatically get access to an ARPES spectrum as an xr.DataArray."""
import xarray as xr
from arpes.typing import DataType

__all__ = (
    "normalize_to_spectrum",
    "normalize_to_dataset",
)


def normalize_to_spectrum(data: DataType):
    """Tries to extract the actual ARPES spectrum from a dataset containing other variables."""
    from arpes.io import load_data
    import arpes.xarray_extensions

    if isinstance(data, xr.Dataset):
        if "up" in data.data_vars:
            return data.up

        return data.S.spectrum

    if isinstance(data, str):
        return normalize_to_spectrum(load_data(data))

    # not guaranteed to be a spectrum, but close enough
    return data


def normalize_to_dataset(data: DataType):
    """Loads data if we were given a path instead of a loaded data sample."""
    from arpes.io import load_data

    if isinstance(data, xr.Dataset):
        return data

    if isinstance(data, (str, int)):
        return load_data(data)
