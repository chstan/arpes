import numpy as np

from arpes.provenance import update_provenance
from arpes.typing import DataType
from utilities import normalize_to_spectrum

__all__ = ('find_t0', 'relative_change')

@update_provenance('Created simple subtraction map')
def relative_change(data: DataType, t0=None):
    spectrum = normalize_to_spectrum(data)

    delay_coords = spectrum.coords['delay']
    scan_start = np.min(delay_coords)

    before_t0 = spectrum.sel(delay=slice(None, 0))
    subtracted = spectrum - before_t0.mean('delay')
    subtracted.attrs['subtracted'] = True
    return subtracted


def find_t0(data: DataType, e_bound=0.05):
    """
    Attempts to find the effective t0 in a spectrum by fitting a peak to the counts that occur
    far enough above e_F
    :param data:
    :param e_bound: Lower bound on the energy to use for the fitting
    :return:
    """
    spectrum = data.S.spectrum
    assert('delay' in spectrum.dims)

    sum_dims = set(spectrum.dims)
    sum_dims.remove('delay')
    return spectrum.sel(eV=slice(e_bound, None)).sum(list(sum_dims))
