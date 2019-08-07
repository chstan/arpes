"""
.. module:: gap
   :synopsis: Utilities for gap fitting in ARPES, contains tools to normalize by Fermi-Dirac occupation

.. moduleauthor:: Conrad Stansbury <chstan@berkeley.edu>
"""
import warnings

import numpy as np

import xarray as xr
from arpes.constants import K_BOLTZMANN_EV_KELVIN
from arpes.fits.fit_models import AffineBroadenedFD
from arpes.provenance import update_provenance
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ('normalize_by_fermi_dirac', 'determine_broadened_fermi_distribution',
           'symmetrize')


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


@update_provenance('Normalize By Fermi Dirac')
def normalize_by_fermi_dirac(data: DataType, reference_data: DataType = None, plot=False,
                             broadening=None,
                             temperature_axis=None,
                             temp_offset=0, **kwargs):
    """
    Normalizes data according to a Fermi level reference on separate data or using the same source spectrum.

    To do this, a linear density of states is multiplied against a resolution broadened Fermi-Dirac
    distribution (`arpes.fits.fit_models.AffineBroadenedFD`). We then set the density of states to 1 and
    evaluate this model to obtain a reference that the desired spectrum is normalized by.

    :param data: Data to be normalized.
    :param reference_data: A reference spectrum, typically a metal reference. If not provided the
                           integrated data is used. Beware: this is inappropriate if your data is gapped.
    :param plot: A debug flag, allowing you to view the normalization spectrum and relevant curve-fits.
    :param broadening: Detector broadening.
    :param temperature_axis: Temperature coordinate, used to adjust the quality
                             of the reference for temperature dependent data.
    :param temp_offset: Temperature calibration in the case of low temperature data. Useful if the
                        temperature at the sample is known to be hotter than the value recorded off of a diode.
    :param kwargs:
    :return:
    """
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


def _shift_energy_interpolate(data: DataType, shift=None):
    if shift is not None:
        pass
        # raise NotImplementedError("arbitrary shift not yet implemented")

    data = normalize_to_spectrum(data).S.transpose_to_front('eV')

    new_data = data.copy(deep=True)
    new_axis = new_data.coords['eV']
    new_values = new_data.values * 0

    if shift is None:
        closest_to_zero = data.coords['eV'].sel(eV=0, method='nearest')
        shift = -closest_to_zero

    stride = data.T.stride('eV', generic_dim_names=False)

    if np.abs(shift) >= stride:
        n_strides = int(shift / stride)
        new_axis = new_axis + n_strides * stride

        shift = shift - stride * n_strides

    new_axis = new_axis + shift

    weight = float(shift / stride)

    new_values = new_values + data.values * (1 - weight)
    if shift > 0:
        new_values[1:] = new_values[1:] + data.values[:-1] * weight
    if shift < 0:
        new_values[:-1] = new_values[:-1] + data.values[1:] * weight

    new_data.coords['eV'] = new_axis
    new_data.values = new_values

    return new_data


@update_provenance('Symmetrize')
def symmetrize(data: DataType, subpixel=False, full_spectrum=False):
    """
    Symmetrizes data across the chemical potential. This provides a crude tool by which
    gap analysis can be performed. In this implementation, subpixel accuracy is achieved by
    interpolating data.

    :param data: Input array.
    :param subpixel: Enable subpixel correction
    :param full_spectrum: Returns data above and below the chemical potential. By default, only
           the bound part of the spectrum (below the chemical potential) is returned, because
           the other half is identical.
    :return:
    """
    data = normalize_to_spectrum(data).S.transpose_to_front('eV')

    if subpixel or full_spectrum:
        data = _shift_energy_interpolate(data)

    above = data.sel(eV=slice(0, None))
    below = data.sel(eV=slice(None, 0)).copy(deep=True)

    l = len(above.coords['eV'])

    zeros = below.values * 0
    zeros[-l:] = above.values[::-1]

    below.values = below.values + zeros

    if full_spectrum:
        if not subpixel:
            warnings.warn("full spectrum symmetrization uses subpixel correction")

        full_data = below.copy(deep=True)

        new_above = full_data.copy(deep=True)[::-1]
        new_above.coords['eV'] = (new_above.coords['eV'] * -1)

        full_data = xr.concat([full_data, new_above[1:]], dim='eV')

        result = full_data
    else:
        result = below

    return result
