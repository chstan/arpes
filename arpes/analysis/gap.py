"""
.. module:: gap
   :synopsis: Utilities for gap fitting in ARPES, contains tools to normalize by Fermi-Dirac occupation

.. moduleauthor:: Conrad Stansbury <chstan@berkeley.edu>
"""
import lmfit as lf
import numpy as np
from lmfit.models import update_param_vals
from scipy.ndimage import gaussian_filter

from arpes.typing import DataType
from arpes.fits import XModelMixin
from arpes.constants import K_BOLTZMANN_EV_KELVIN
from arpes.utilities import normalize_to_spectrum

__all__ = ('normalize_by_fermi_dirac', 'determine_broadened_fermi_distribution', 'AffineBroadenedFD',
           'symmetrize')

def affine_broadened_fd(x, fd_center=0, fd_width=0.003, conv_width=0.02, const_bkg=1, lin_bkg=0, offset=0):
    """
    Fermi function convoled with a Gaussian together with affine background
    :param x: value to evaluate function at
    :param center: center of the step
    :param width: width of the step
    :param erf_amp: height of the step
    :param lin_bkg: linear background slope
    :param const_bkg: constant background
    :return:
    """
    dx = x - fd_center
    x_scaling = x[1] - x[0]
    fermi = 1 / (np.exp(dx / fd_width) + 1)
    return gaussian_filter(
        (const_bkg + lin_bkg * dx) * fermi,
        sigma=conv_width / x_scaling
    ) + offset


class AffineBroadenedFD(XModelMixin):
    """
    A model for fitting an affine density of states with resolution broadened Fermi-Dirac occupation
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(affine_broadened_fd, **kwargs)

        self.set_param_hint('offset', min=0.)
        self.set_param_hint('fd_width', min=0.)
        self.set_param_hint('conv_width', min=0.)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%sfd_center' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.mean().item() * 2)
        pars['%soffset' % self.prefix].set(value=data.min().item())

        pars['%sfd_width' % self.prefix].set(0.005)  # TODO we can do better than this
        pars['%sconv_width' % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC



def determine_broadened_fermi_distribution(reference_data: DataType, fixed_temperature=True):
    """
    Determine the parameters for broadening by temperature and instrumental resolution
    for a piece of data.

    As a general rule, we first try to estimate the instrumental broadening and linewidth broadening
    according to calibrations provided for the beamline + instrument, as a starting point.

    We also calculate the thermal broadening to expect, and fit an edge location. Then we use a Gaussian
    convolved Fermi-Dirac distribution against an affine density of states near the Fermi level, with a constant
    offset background above the Fermi level as a simple but effective model when away from lineshapes.

    These parameters can be used to bootstrap a fit to actual data or used directly in ``normalize_by_fermi_dirac``.


    :param reference_data:
    :return:
    """
    params = {}

    if fixed_temperature:
        params['fd_width'] = {
            'value': reference_data.S.temp * K_BOLTZMANN_EV_KELVIN,
            'vary': False,
        }

    reference_data = normalize_to_spectrum(reference_data)

    sum_dims = list(reference_data.dims)
    sum_dims.remove('eV')

    return AffineBroadenedFD().guess_fit(reference_data.sum(sum_dims), params=params)


def normalize_by_fermi_dirac(data: DataType, reference_data: DataType=None, plot=False,
                             broadening=None,
                             temperature_axis=None,
                             temp_offset=0, **kwargs):
    reference_data = data if reference_data is None else reference_data
    broadening_fit = determine_broadened_fermi_distribution(reference_data, **kwargs)
    broadening = broadening_fit.params['conv_width'].value if broadening is None else broadening

    if plot:
        print('Gaussian broadening is: {} meV (Gaussian sigma)'.format(
            broadening_fit.params['conv_width'].value * 1000))
        print('Fermi edge location is: {} meV (fit chemical potential)'.format(
            broadening_fit.params['fd_center'].value * 1000))
        print('Fermi width is: {} meV (fit fermi width)'.format(
            broadening_fit.params['fd_width'].value * 1000))

        broadening_fit.plot()

    offset = broadening_fit.params['offset'].value
    without_offset = broadening_fit.eval(offset=0)

    cut_index = -np.argmax(without_offset[::-1] > 0.1 * offset)
    cut_energy = reference_data.coords['eV'].values[cut_index]

    if temperature_axis is None and 'temp' in data.dims:
        temperature_axis = 'temp'

    transpose_order = list(data.dims)
    transpose_order.remove('eV')

    if temperature_axis:
        transpose_order.remove(temperature_axis)
        transpose_order = transpose_order + [temperature_axis]

    transpose_order = transpose_order + ['eV']

    without_background = (data - data.sel(eV=slice(cut_energy, None)).mean('eV')).transpose(*transpose_order)

    if temperature_axis:
        without_background = normalize_to_spectrum(without_background)
        divided = without_background.T.map_axes(
            temperature_axis, lambda x, coord: x / broadening_fit.eval(
                x=x.coords['eV'].values, lin_bkg=0, const_bkg=1, offset=0,
                conv_width=broadening,
                fd_width=(coord[temperature_axis] + temp_offset) * K_BOLTZMANN_EV_KELVIN))
    else:
        without_background = normalize_to_spectrum(without_background)
        divided = without_background / broadening_fit.eval(
            x=data.coords['eV'].values,
            conv_width=broadening,
            lin_bkg=0, const_bkg=1, offset=0)

    divided.coords['eV'].values = divided.coords['eV'].values - broadening_fit.params['fd_center'].value
    return divided


def symmetrize(data: DataType):
    data = normalize_to_spectrum(data).S.transpose_to_front('eV')

    above = data.sel(eV=slice(0, None))
    below = data.sel(eV=slice(None, 0)).copy(deep=True)

    l = len(above.coords['eV'])

    zeros = below.values * 0
    zeros[-l:] = above.values[::-1]

    below.values = below.values + zeros

    return below