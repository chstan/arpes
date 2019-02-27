"""
.. module:: gap
   :synopsis: Utilities for gap fitting in ARPES, contains tools to normalize by Fermi-Dirac occupation

.. moduleauthor:: Conrad Stansbury <chstan@berkeley.edu>
"""
import numpy as np

from arpes.typing import DataType
from arpes.constants import K_BOLTZMANN_EV_KELVIN
from arpes.utilities import normalize_to_spectrum
from arpes.fits.fit_models import AffineBroadenedFD

import xarray as xr
import warnings

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

"""
def symmetrize(data: DataType):
    data = normalize_to_spectrum(data).S.transpose_to_front('eV')

    above = data.sel(eV=slice(0, None))
    below = data.sel(eV=slice(None, 0)).copy(deep=True)

    l = len(above.coords['eV'])

    zeros = below.values * 0
    print(zeros.shape)
    zeros[-l:] = above.values[::-1]

    below.values = below.values + zeros

    return below
"""

def _shift_energy_interpolate(data: DataType,shift=None):
    if shift is not None:
        pass
        # raise NotImplementedError("arbitrary shift not yet implemented")
        
    data = normalize_to_spectrum(data).S.transpose_to_front('eV')
    
    new_data = data.copy(deep=True)
    new_axis = new_data.coords['eV']
    new_values = new_data.values * 0
    
    if shift is None:
        closest_to_zero = data.coords['eV'].sel(eV=0,method='nearest')
        shift = -closest_to_zero
    
    stride = data.T.stride('eV',generic_dim_names=False)
    
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

def symmetrize(data: DataType,subpixel=False,full_spectrum=False):
    data = normalize_to_spectrum(data).S.transpose_to_front('eV')
    
    if subpixel or full_spectrum:
        data = _shift_energy_interpolate(data)

    new_data = data * 0
    
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

        full_data = xr.concat([full_data,new_above[1:]],dim='eV')
        
        result = full_data
    else:
        result = below
        
    return result
