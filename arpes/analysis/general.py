import numpy as np
import xarray as xr

import arpes.models.band
import arpes.utilities
import arpes.utilities.math
from arpes.provenance import update_provenance
from .filter import gaussian_filter_arr

__all__ = ('normalize_by_fermi_distribution', 'symmetrize_axis')


@update_provenance('Normalized by the 1/Fermi Dirac Distribution at sample temp')
def normalize_by_fermi_distribution(data, max_gain=None, rigid_shift=0, instrumental_broadening=None):
    """
    Normalizes a scan by 1/the fermi dirac distribution. You can control the maximum gain with ``clamp``, and whether
    the Fermi edge needs to be shifted (this is for those desperate situations where you want something that
    "just works") via ``rigid_shift``.

    :param data: Input
    :param clamp: Maximum value for the gain. By default the value used is the mean of the spectrum.
    :param rigid_shift: How much to shift the spectrum chemical potential.
    Pass the nominal value for the chemical potential in the scan. I.e. if the chemical potential is at BE=0.1, pass
    rigid_shift=0.1.
    :param instrumental_broadening: Instrumental broadening to use for convolving the distribution
    :return: Normalized DataArray
    """
    distrib = arpes.utilities.math.fermi_distribution(data.coords['eV'].values - rigid_shift, data.S.temp)

    # don't boost by more than 90th percentile of input, by default
    if max_gain is None:
        max_gain = np.mean(data.values)

    distrib[distrib < 1/max_gain] = 1/max_gain
    distrib_arr = xr.DataArray(
        distrib,
        {'eV': data.coords['eV'].values},
        ['eV']
    )

    if instrumental_broadening is not None:
        distrib_arr = gaussian_filter_arr(distrib_arr, sigma={'eV': instrumental_broadening})

    return data / distrib_arr


#@update_provenance('Symmetrize axis')
def symmetrize_axis(data, axis_name, flip_axes=None):
    selector = {}
    selector[axis_name] = slice(None, None, -1)
    rev = data.sel(**selector).copy()
    rev.coords[axis_name].values = -rev.coords[axis_name].values

    if flip_axes is None:
        flip_axes = []

    for axis in flip_axes:
        selector = {}
        selector[axis] = slice(None, None, -1)
        rev = rev.sel(**selector)
        rev.coords[axis].values = -rev.coords[axis].values

    return rev.combine_first(data)
