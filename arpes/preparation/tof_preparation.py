import math

import numpy as np
import xarray as xr

from .axis_preparation import transform_dataarray_axis

__all__ = ['build_KE_coords_to_time_pixel_coords', 'build_KE_coords_to_time_coords', 'process_DLD']


def convert_to_kinetic_energy(dataarray, kinetic_energy_axis):
    """
    Convert the ToF timing information into an energy histogram

    The core of these routines come from the Igor procedures in
    ``LoadTOF7_3.51.ipf``.

    To explain in a bit more detail what actaully happens, this
    function essentially

    1. Determines the time to energy conversion factor
    2. Calculates the requested energy binning range and allocates arrays
    3. Rebins a time spectrum into an energy spectrum, preserving the
       spectral weight, this requires a modicum of care around splitting
       counts at the edges of the new bins.
    """

    # This should be simplified
    # c = (0.5) * (9.11e-31) * self.mstar * (self.length ** 2) / (1.6e-19) * (1e18)
    # Removed factors of ten and substituted mstar = 0.5
    c = (0.5) * (9.11e6) * 0.5 * (dataarray.S.spectrometer['length'] ** 2) / 1.6

    new_dim_order = list(dataarray.dims)
    new_dim_order.remove('time')
    new_dim_order = ['time'] + new_dim_order
    dataarray.transpose(*new_dim_order)
    new_dim_order[0] = 'KE'

    timing = dataarray.coords['time']
    assert(timing[1] > timing[0])
    t_min, t_max = np.min(timing), np.max(timing)

    # Prep arrays
    #self.energies = np.linspace(E_min_R, E_max_R - self.dE, NE)
    d_energy = kinetic_energy_axis[1] - kinetic_energy_axis[0]
    time_index = 0 # after transpose
    new_shape = list(dataarray.data.shape)
    new_shape[time_index] = len(kinetic_energy_axis)

    new_data = np.zeros(tuple(new_shape))

    def energy_to_time(conv, energy):
        return math.sqrt(conv / energy)

    # Rebin data
    old_data = dataarray.data
    for i, E in enumerate(kinetic_energy_axis):
        t_L = energy_to_time(c, E + d_energy / 2)
        t_S = energy_to_time(c, E - d_energy / 2)

        # clamp
        t_L = t_L if t_L <= t_max else t_max
        t_S = t_S if t_S > t_min else t_min

        # with some math we could back calculate the index, but we don't need to
        t_L_idx = np.searchsorted(timing, t_L)
        t_S_idx = np.searchsorted(timing, t_S)

        new_data[i] = np.sum(old_data[t_L_idx:t_S_idx], axis=0) + \
                      ((timing[t_L_idx] - t_L) * old_data[t_L_idx]) + \
                      ((t_S - timing[t_S_idx - 1]) * old_data[t_S_idx - 1]) / d_energy

    new_coords = dict(dataarray.coords)
    del new_coords['time']
    new_coords['KE'] = kinetic_energy_axis

    # Put provenance here

    return xr.DataArray(
        new_data,
        coords=new_coords,
        dims=new_dim_order,
        attrs=dataarray.attrs,
        name=dataarray.name
    )


def build_KE_coords_to_time_pixel_coords(dataset: xr.Dataset, interpolation_axis):
    conv = (0.5) * (9.11e6) * 0.5 * (dataset.S.spectrometer['length'] ** 2) / 1.6
    time_res = 0.17 # this is only approximate

    def KE_coords_to_time_pixel_coords(coords, axis=None):
        """
        Like ``KE_coords_to_time_coords`` but it converts to the raw timing pixels off of
        a DLD instead to the unitful values that we receive from the Spin-ToF DAQ
        :param coords: tuple of coordinates
        :return: new tuple of converted coordinates
        """
        kinetic_energy_pixel = coords[axis]
        kinetic_energy = interpolation_axis[kinetic_energy_pixel]
        real_timing = math.sqrt(conv / kinetic_energy)
        pixel_timing = (real_timing - dataset.attrs['timing_offset']) / time_res
        coords_list = list(coords)
        coords_list[axis] = pixel_timing

        return tuple(coords_list)
    return KE_coords_to_time_pixel_coords


def build_KE_coords_to_time_coords(dataset: xr.Dataset, interpolation_axis):
    conv = (0.5) * (9.11e6) * 0.5 * (dataset.S.spectrometer['length'] ** 2) / 1.6

    def KE_coords_to_time_coords(coords, axis=None):
        """
        Used to convert the timing coordinates off of the spin-ToF to kinetic energy coordinates.
        As the name suggests, because of how scipy.ndimage interpolates, we require the inverse
        coordinate transform.

        All coordinates except for the energy coordinate are left untouched.
        :param coords: tuple of coordinates
        :return: new tuple of converted coordinates
        """

        kinetic_energy = interpolation_axis[coords[axis]]
        real_timing = math.sqrt(conv / kinetic_energy)
        real_timing = real_timing# - dataset.attrs['timing_offset']
        coords_list = list(coords)
        coords_list[axis] = real_timing
        return tuple(coords_list)

    return KE_coords_to_time_coords


def convert_SToF_to_energy(dataset: xr.Dataset):
    e_min, e_max = 0.1, 10.

    # TODO, we can better infer a reasonable gridding here
    spacing = 0.005 # 5meV gridding
    ke_axis = np.linspace(e_min, e_max, int((e_max - e_min) / spacing))

    drs = {k: v for k, v in dataset.data_vars.items() if 'time' in v.dims}

    new_dataarrays = [convert_to_kinetic_energy(dr, ke_axis) for dr in drs.values()]

    for k in drs.keys():
        old_dr = dataset[k]
        del dataset[k]
        dataset[old_dr.name + '_time'] = old_dr.rename(old_dr.name + '_time')

    for v in new_dataarrays:
        dataset[v.name] = v

    return dataset


def process_DLD(dataset: xr.Dataset):
    e_min = 1
    ke_axis = np.linspace(e_min, dataset.attrs['E_max'], (dataset.attrs['E_max'] - e_min) / dataset.attrs['dE'])
    dataset = transform_dataarray_axis(
        build_KE_coords_to_time_pixel_coords(dataset, ke_axis),
        't_pixels', 'KE', ke_axis, dataset, lambda x: 'KE_spectrum'
    )

    return dataset