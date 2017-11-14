import math

import numpy as np
import xarray as xr

from arpes.utilities import get_spectrometer
from .axis_preparation import transform_dataarray_axis

__all__ = ['build_KE_coords_to_time_pixel_coords', 'build_KE_coords_to_time_coords', 'process_DLD']


def build_KE_coords_to_time_pixel_coords(dataset: xr.Dataset, interpolation_axis):
    t_pixel_axis = dataset.raw.dims.index('t_pixels')

    conv = (0.5) * (9.11e6) * 0.5 * (get_spectrometer(dataset)['length'] ** 2) / 1.6
    time_res = 0.17 # this is only approximate

    def KE_coords_to_time_pixel_coords(coords):
        """
        Like ``KE_coords_to_time_coords`` but it converts to the raw timing pixels off of
        a DLD instead to the unitful values that we receive from the Spin-ToF DAQ
        :param coords: tuple of coordinates
        :return: new tuple of converted coordinates
        """
        kinetic_energy_pixel = coords[t_pixel_axis]
        kinetic_energy = interpolation_axis[kinetic_energy_pixel]
        real_timing = math.sqrt(conv / kinetic_energy)
        pixel_timing = (real_timing - dataset.attrs['timing_offset']) / time_res
        coords_list = list(coords)
        coords_list[t_pixel_axis] = pixel_timing

        return tuple(coords_list)
    return KE_coords_to_time_pixel_coords


def build_KE_coords_to_time_coords(dataset: xr.Dataset):
    t_axis = dataset.raw.dims.index('t')

    conv = (0.5) * (9.11e6) * 0.5 * (get_spectrometer(dataset)['length'] ** 2) / 1.6

    def KE_coords_to_time_coords(coords):
        """
        Used to convert the timing coordinates off of the spin-ToF to kinetic energy coordinates.
        As the name suggests, because of how scipy.ndimage interpolates, we require the inverse
        coordinate transform.

        All coordinates except for the energy coordinate are left untouched.
        :param coords: tuple of coordinates
        :return: new tuple of converted coordinates
        """
        kinetic_energy = coords[t_axis]
        real_timing = math.sqrt(conv / kinetic_energy)
        coords_list = list(coords)
        coords_list[t_axis] = real_timing

        return coords

    return KE_coords_to_time_coords


def process_DLD(dataset: xr.Dataset):
    e_min = 1
    ke_axis = np.linspace(e_min, dataset.attrs['E_max'], (dataset.attrs['E_max'] - e_min) / dataset.attrs['dE'])
    dataset = transform_dataarray_axis(
        build_KE_coords_to_time_pixel_coords(dataset, ke_axis),
        't_pixels', 'KE', ke_axis, dataset, 'KE_spectrum'
    )

    return dataset