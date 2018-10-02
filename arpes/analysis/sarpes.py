import xarray as xr

from arpes.typing import DataType
from utilities import normalize_to_dataset
from utilities.math import polarization

__all__ = ('to_intensity_polarization',)


def to_intensity_polarization(data: DataType):
    data = normalize_to_dataset(data)

    assert('up' in data.data_vars and 'down' in data.data_vars)

    intensity = data.up + data.down
    spectrum_polarization = polarization(data.up, data.down)

    return xr.Dataset({
        'intensity': intensity,
        'polarization': spectrum_polarization
    })