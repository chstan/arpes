from arpes.analysis import rebin
from arpes.typing import DataType
from arpes.utilities.normalize import normalize_to_spectrum
from arpes.fits import broadcast_model, GStepBModel

__all__ = ('fs_gap',)


def fs_gap(data: DataType, shape=None, energy_range=None):
    data = normalize_to_spectrum(data)

    if energy_range is None:
        energy_range = slice(-0.1, None)

    data.sel(eV=energy_range)

    reduction=None
    if shape is None:
        # Just rebin the data along 'phi'
        reduction = {'phi': 16}

    data = rebin(data, reduction=reduction, shape=shape)
    return broadcast_model(GStepBModel, data, ['phi', 'beta'])
