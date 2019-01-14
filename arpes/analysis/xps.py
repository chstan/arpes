import numpy as np

from arpes.analysis.savitzky_golay import *
from arpes.analysis.general import rebin
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ('approximate_core_levels',)


def local_minima(a, promenance=3):
    conditions = a == a
    for i in range(1, promenance + 1):
        current_conditions = np.r_[[False] * i, a[i:] < a[:-i]] & np.r_[a[:-i] < a[i:], [False] * i]
        conditions = conditions & current_conditions

    return conditions


def local_maxima(a, promenance=3):
    return local_minima(-a, promenance)


def approximate_core_levels(data: DataType, window_size=None, order=5, binning=3, promenance=5):
    data = normalize_to_spectrum(data)

    dos = data.S.sum_other(['eV']).sel(eV=slice(None, -20))

    if window_size is None:
        window_size = int(len(dos) / 40) # empirical, may change
        if window_size % 2 == 0:
            window_size += 1

    smoothed = rebin(savitzky_golay(dos, window_size, order), eV=binning)

    indices = np.argwhere(local_maxima(smoothed.values, promenance=promenance))
    energies = [smoothed.coords['eV'][idx].item() for idx in indices]

    return energies
