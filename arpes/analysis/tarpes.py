import numpy as np

from arpes.provenance import update_provenance
from arpes.typing import DataType
from arpes.preparation import normalize_dim
from arpes.utilities import normalize_to_spectrum

__all__ = ('find_t0', 'relative_change', 'normalized_relative_change')


@update_provenance('Normalized subtraction map')
def normalized_relative_change(data: DataType, t0=None, buffer=0.3, normalize_delay=True):
    spectrum = normalize_to_spectrum(data)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, 'delay')
    subtracted = relative_change(spectrum, t0, buffer, normalize_delay=False)
    normalized = subtracted / spectrum
    normalized.values[np.isinf(normalized.values)] = 0
    normalized.values[np.isnan(normalized.values)] = 0
    return normalized


@update_provenance('Created simple subtraction map')
def relative_change(data: DataType, t0=None, buffer=0.3, normalize_delay=True):
    spectrum = normalize_to_spectrum(data)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, 'delay')

    delay_coords = spectrum.coords['delay']
    delay_start = np.min(delay_coords)

    if t0 is None:
        t0 = spectrum.S.t0 or find_t0(spectrum)

    assert(t0 - buffer > delay_start)

    before_t0 = spectrum.sel(delay=slice(None, t0 - buffer))
    subtracted = spectrum - before_t0.mean('delay')
    return subtracted


def find_t0(data: DataType, e_bound=0.02, approx=True):
    """
    Attempts to find the effective t0 in a spectrum by fitting a peak to the counts that occur
    far enough above e_F
    :param data:
    :param e_bound: Lower bound on the energy to use for the fitting
    :return:
    """
    spectrum = normalize_to_spectrum(data)

    assert('delay' in spectrum.dims)
    assert('eV' in spectrum.dims)

    sum_dims = set(spectrum.dims)
    sum_dims.remove('delay')
    sum_dims.remove('eV')

    summed = spectrum.sum(list(sum_dims)).sel(eV=slice(e_bound, None)).mean('eV')
    coord_max = summed.argmax().item()

    return summed.coords['delay'].values[coord_max]

