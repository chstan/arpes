import xarray as xr

from arpes.typing import DataType

__all__ = ('normalize_to_spectrum', 'normalize_to_dataset',)


def normalize_to_spectrum(data: DataType):
    from arpes.io import load_dataset
    if isinstance(data, xr.Dataset):
        if 'up' in data.data_vars:
            return data.up

        return data.S.spectrum

    if isinstance(data, str):
        return normalize_to_spectrum(load_dataset(dataset_uuid=data))

    # not guaranteed to be a spectrum, but close enough
    return data


def normalize_to_dataset(data: DataType):
    from arpes.io import load_dataset
    if isinstance(data, xr.Dataset):
        return data

    if isinstance(data, str):
        return load_dataset(dataset_uuid=data)
