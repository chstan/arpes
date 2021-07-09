"""X-ray photoelectron spectroscopy related analysis.

Primarily, curve fitting and peak-finding utilities for XPS.
"""

import numpy as np

from arpes.analysis.general import rebin
from arpes.analysis.savitzky_golay import savitzky_golay
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ("approximate_core_levels",)


def local_minima(a: np.ndarray, promenance: int = 3) -> np.ndarray:
    """Calculates local minima (maxima) according to a prominence criterion.

    The point should be lower than any in the region around it.

    Rather than searching manually, we perform some fancy indexing to do the calculation
    across the whole array simultaneously and iterating over the promenance criterion instead of
    through the data and the promenance criterion.

    Args:
        a: The input array to calculate local minima over
        promenance: The prominence over indices required to be called a local minimum

    Returns:
        A mask where the local minima are True and other values are False.
    """
    conditions = a == a
    for i in range(1, promenance + 1):
        current_conditions = np.r_[[False] * i, a[i:] < a[:-i]] & np.r_[a[:-i] < a[i:], [False] * i]
        conditions = conditions & current_conditions

    return conditions


def local_maxima(a, promenance=3):
    return local_minima(-a, promenance)


local_maxima.__doc__ = local_minima.__doc__


def approximate_core_levels(data: DataType, window_size=None, order=5, binning=3, promenance=5):
    """Approximately locates core levels in a spectrum.

    Data is first smoothed, and then local maxima with sufficient prominence over
    other nearby points are selected as peaks.

    This can be helfpul to "seed" a curve fitting analysis for XPS.

    Args:
        data: An XPS spectrum.
        window_size: Savitzky-Golay window size
        order: Savitzky-Golay order
        binning: Used for approximate smoothing
        promenance: Required promenance over nearby peaks

    Returns:
        A set of energies with candidate peaks.
    """
    data = normalize_to_spectrum(data)

    dos = data.S.sum_other(["eV"]).sel(eV=slice(None, -20))

    if window_size is None:
        window_size = int(len(dos) / 40)  # empirical, may change
        if window_size % 2 == 0:
            window_size += 1

    smoothed = rebin(savitzky_golay(dos, window_size, order), eV=binning)

    indices = np.argwhere(local_maxima(smoothed.values, promenance=promenance))
    energies = [smoothed.coords["eV"][idx].item() for idx in indices]

    return energies
