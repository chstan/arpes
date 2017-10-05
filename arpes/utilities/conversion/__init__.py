"""
Helper functions for coordinate transformations. All the functions here
assume standard polar angles, for better or worse, so you might need to
massage your inputs slightly in order to get them into an appropriate form.

Functions here must accept constants or numpy arrays as valid inputs,
so all standard math functions have been replaced by their equivalents out
of numpy. Array broadcasting should handle any issues or weirdnesses that
would encourage the use of direct iteration, but in case you need to write
a conversion directly, be aware that any functions here must work on arrays
as well for consistency with client code.

Through the code that follows, there are some conventions on the names of angles
to make the code easier to follow:

Standard vertically oriented cryostats:

'polar' is the name of the angle that describes rotation around \hat{z}
'beta' is the name of the angle that describes rotation around \hat{x}
'sample_phi' is the name of the angle that describes rotation around the sample normal
'phi' is the name of the angle that describes the angle along the analyzer entrance slit

Additionally, everywhere, 'eV' denotes binding energies. Other energy units should
be labelled as:

Kinetic energy -> 'kinetic_energy'
Binding energy -> 'binding_energy'
Photon energy -> 'hv'

Other angles:
Sample elevation/tilt/beta angle -> 'polar'
Analyzer polar angle -> 'phi'
"""

# pylint: disable=W0613, C0103

import itertools
from copy import deepcopy

import numpy as np
import scipy.interpolate
import xarray as xr

import arpes
import arpes.constants as consts
from arpes.exceptions import UnimplementedException
from arpes.provenance import provenance
from arpes.utilities import phi_offset, polar_offset, inner_potential, work_function, photon_energy

# TODO Add conversion utilities for MC (i.e. vertical slit)
# TODO Add conversion utilities for photon energy dependence
# TODO Add conversion utilities that work for lower dimensionality, i.e. the ToF
# TODO Add provenance capabilities


K_SPACE_BORDER = 0.1


class CoordinateConverter(object):
    def __init__(self, arr: xr.DataArray, *args, **kwargs):
        # Intern the volume so that we can check on things during computation
        self.arr = arr

    def prep(self, arr: xr.DataArray):
        """
        The CoordinateConverter.prep method allows you to pre-compute some transformations
        that are common to the individual coordinate transform methods as an optimization.

        This is useful if you want the conversion methods to have separation of concern,
        but if it is advantageous for them to be able to share a computation of some
        variable. An example of this is in BE-kx-ky conversion, where computing k_p_tot
        is a step in both converting kx and ky, so we would like to avoid doing it twice.

        Of course, you can neglect this function entirely. Another technique is to simple
        cache computations as they arrive. This is the technique that is used in
        ConvertKxKy below
        :param arr:
        :return:
        """
        pass

    def kspace_to_BE(self, binding_energy, *args, **kwargs):
        return binding_energy

    def conversion_for(self, dim):
        pass

    def get_coordinates(self, resolution: dict=None):
        coordinates = {}
        coordinates['eV'] = self.arr.coords['eV']
        return coordinates

class ConvertKpKzV0(CoordinateConverter):
    # TODO implement
    def __init__(self, *args, **kwargs):
        super(ConvertKpKzV0, self).__init__(*args, **kwargs)

class ConvertKp(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super(ConvertKp, self).__init__(*args, **kwargs)
        self.k_tot = None

    def get_coordinates(self, resolution: dict=None):
        if resolution is None:
            resolution = {}

        coordinates = super(ConvertKp, self).get_coordinates(resolution)
        (kp_low, kp_high) = calculate_kp_bounds(self.arr)

        coordinates['kp'] = np.arange(kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER,
                                      resolution.get('kp', arpes.constants.MEDIUM_FINE_K_GRAINING))

        return coordinates

    def compute_k_tot(self, binding_energy):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            photon_energy(self.arr) - work_function(self.arr) + binding_energy)

    def kspace_to_phi(self, binding_energy, kp, *args, **kwargs):
        polar_angle = self.arr.attrs.get('polar')
        polar_angle -= polar_offset(self.arr)

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        return (180 / np.pi) * np.arcsin(kp / self.k_tot / np.cos(
            polar_angle * np.pi / 180)) + phi_offset(self.arr)


    def conversion_for(self, dim):
        return {
            'eV': self.kspace_to_BE,
            'phi': self.kspace_to_phi
        }.get(dim, None)

class ConvertKpKz(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super(ConvertKpKz, self).__init__(*args, **kwargs)
        self.hv = None

    def get_coordinates(self, resolution: dict=None):
        if resolution is None:
            resolution = {}

        coordinates = super(ConvertKpKz, self).get_coordinates(resolution)

        ((kp_low, kp_high), (kz_low, kz_high)) = calculate_kp_kz_bounds(self.arr)

        coordinates['kp'] = np.arange(kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER,
                                      resolution.get('kp', arpes.constants.MEDIUM_FINE_K_GRAINING))
        coordinates['kz'] = np.arange(kz_low - K_SPACE_BORDER, kz_high + K_SPACE_BORDER,
                                      resolution.get('kz', arpes.constants.MEDIUM_FINE_K_GRAINING))

        return coordinates

    def kspace_to_hv(self, binding_energy, kp, kz, *args, **kwargs):
        # x = kp, y = kz, z = BE
        if self.hv is None:
            inner_v = inner_potential(self.arr)
            wf = work_function(self.arr)
            self.hv = arpes.constants.HV_CONVERSION * (kp ** 2 + kz ** 2) + (
                -inner_v - binding_energy + wf)

        return self.hv

    def kspace_to_phi(self, binding_energy, kp, kz, *args, **kwargs):
        if self.hv is None:
            self.kspace_to_hv(binding_energy, kp, kz, *args, **kwargs)

        def kp_to_polar(kinetic_energy_out, kp):
            return 180 / np.pi * np.arcsin(kp / (arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy_out)))

        return kp_to_polar(self.hv + work_function(self.arr), kp) + phi_offset(self.arr)

    def conversion_for(self, dim):
        return {
            'eV': self.kspace_to_BE,
            'hv': self.kspace_to_hv,
            'phi': self.kspace_to_phi,
        }.get(dim, None)


class ConvertKxKyKz(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super(ConvertKxKyKz, self).__init__(*args, **kwargs)


class ConvertKxKy(CoordinateConverter):
    # TODO Convert all angles to radians where possible, avoid computations back and forth
    def __init__(self, *args, **kwargs):
        super(ConvertKxKy, self).__init__(*args, **kwargs)
        self.polar = None
        self.k_tot = None

    def get_coordinates(self, resolution: dict=None):
        if resolution is None:
            resolution = {}

        coordinates = super(ConvertKxKy, self).get_coordinates(resolution)

        ((kx_low, kx_high), (ky_low, ky_high)) = calculate_kx_ky_bounds(self.arr)

        coordinates['kx'] = np.arange(kx_low - K_SPACE_BORDER, kx_high + K_SPACE_BORDER,
                                      resolution.get('kx', arpes.constants.MEDIUM_FINE_K_GRAINING))
        coordinates['ky'] = np.arange(ky_low - K_SPACE_BORDER, ky_high + K_SPACE_BORDER,
                                      resolution.get('ky', arpes.constants.MEDIUM_FINE_K_GRAINING))

        return coordinates

    def compute_k_tot(self, binding_energy):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            photon_energy(self.arr) - work_function(self.arr) + binding_energy)

    def kspace_to_phi(self, binding_energy, kx, ky, *args, **kwargs):
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.polar is None:
            self.kspace_to_polar(binding_energy, kx, ky, *args, **kwargs)

        return (180 / np.pi) * np.arcsin(kx / self.k_tot / np.cos((self.polar - polar_offset(self.arr)) * np.pi / 180)) + \
               phi_offset(self.arr)

    def kspace_to_polar(self, binding_energy, kx, ky, *args, **kwargs):
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        self.polar = (180 / np.pi) * np.arcsin(ky / self.k_tot) + polar_offset(self.arr)
        return self.polar

    def conversion_for(self, dim):
        return {
            'eV': self.kspace_to_BE,
            'phi': self.kspace_to_phi,
            'polar': self.kspace_to_polar,
        }.get(dim, None)


def calculate_kp_kz_bounds(arr: xr.DataArray):
    phi_min = np.min(arr.coords['phi'].values) - phi_offset(arr)
    phi_max = np.max(arr.coords['phi'].values) - phi_offset(arr)

    binding_energy_min, binding_energy_max = np.min(arr.coords['eV'].values), np.max(arr.coords['eV'].values)
    hv_min, hv_max = np.min(arr.coords['hv'].values), np.max(arr.coords['hv'].values)

    def spherical_to_kx(kinetic_energy, theta, phi):
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(
            np.pi / 180 * theta) * np.cos(np.pi / 180 * phi)

    def spherical_to_kz(kinetic_energy, theta, phi, inner_V):
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(
            kinetic_energy * np.cos(np.pi / 180 * theta) ** 2 + inner_V)

    wf = work_function(arr)
    kx_min = min(spherical_to_kx(hv_max - binding_energy_max - wf, phi_min, 0),
                 spherical_to_kx(hv_min - binding_energy_max - wf, phi_min, 0))
    kx_max = max(spherical_to_kx(hv_max - binding_energy_max - wf, phi_max, 0),
                 spherical_to_kx(hv_min - binding_energy_max - wf, phi_max, 0))

    angle_max = max(abs(phi_min), abs(phi_max))
    inner_V = inner_potential(arr)
    kz_min = spherical_to_kz(hv_min + binding_energy_min - wf, angle_max, 0, inner_V)
    kz_max = spherical_to_kz(hv_max + binding_energy_max - wf, 0, 0, inner_V)

    return (
        (round(kx_min, 2), round(kx_max, 2)), # kp
        (round(kz_min, 2), round(kz_max, 2)), # kz
    )


def calculate_kp_bounds(arr: xr.DataArray):
    phi_coords = arr.coords['phi'].values - phi_offset(arr)
    polar = arr.attrs.get('polar') - polar_offset(arr)

    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    phi_mid = (phi_high + phi_low) / 2

    sampled_phi_values = np.array([phi_low, phi_mid, phi_high])

    kinetic_energy = photon_energy(arr) - work_function(arr)
    kps = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(
        np.pi / 180. * sampled_phi_values) * np.cos(np.pi / 180. * polar)

    return round(np.min(kps), 2), round(np.max(kps), 2)

def calculate_kx_ky_bounds(arr: xr.DataArray):
    """
    Calculates the kx and ky range for a dataset with a fixed photon energy

    This is used to infer the gridding that should be used for a k-space conversion.
    Based on Jonathan Denlinger's old codes
    :param arr: Dataset that includes a key indicating the photon energy of the scan
    :return: ((kx_low, kx_high,), (ky_low, ky_high,))
    """
    phi_coords, polar_coords = arr.coords['phi'] - phi_offset(arr), arr.coords['polar'] - polar_offset(arr)

    # Sample hopefully representatively along the edges
    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    polar_low, polar_high = np.min(polar_coords), np.max(polar_coords)
    phi_mid = (phi_high + phi_low) / 2
    polar_mid = (polar_high + polar_low) / 2

    sampled_phi_values = np.array([phi_high, phi_high, phi_mid, phi_low, phi_low,
                                   phi_low, phi_mid, phi_high, phi_high])
    sampled_polar_values = np.array([polar_mid, polar_high, polar_high, polar_high, polar_mid,
                                     polar_low, polar_low, polar_low, polar_mid])
    kinetic_energy = photon_energy(arr) - work_function(arr)

    kxs = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(np.pi / 180. * sampled_phi_values)
    kys = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.cos(
        np.pi / 180. * sampled_phi_values) * np.sin(
        np.pi / 180. * sampled_polar_values)

    return (
        (round(np.min(kxs), 2), round(np.max(kxs), 2)),
        (round(np.min(kys), 2), round(np.max(kys), 2)),
    )


def infer_kspace_coordinate_transform(arr: xr.DataArray):
    """
    Infers appropriate coordinate transform for arr to momentum space.

    This takes into account the extra metadata attached to arr that might be
    useful in inferring the requirements of the coordinate transform, like the
    orientation of the spectrometer slit, and other experimental concerns
    :param arr:
    :return: dict with keys ``target_coordinates``, and a map of the appropriate
    conversion functions
    """
    old_coords = deepcopy(list(arr.coords))
    assert ('eV' in old_coords)
    old_coords.remove('eV')
    old_coords.sort()

    new_coords = {
        ('phi',): ['kp'],
        ('phi', 'polar',): ['kx', 'ky'],
        ('hv', 'phi',): ['kp', 'kz'],
        ('hv', 'phi', 'polar',): ['kx', 'ky', 'kz'],
    }.get(tuple(old_coords))

    # At this point we need to do a bit of work in order to determine the functions
    # that interpolate from k-space back into the recorded variable space

    # TODO Also provide the Jacobian of the coordinate transform to properly
    return {
        'dims': new_coords,
        'transforms': {

        },
        'calculate_bounds': None,
        'jacobian': None,
    }


def grid_interpolator_from_dataarray(arr: xr.DataArray, fill_value=0.0, method='linear',
                                     bounds_error=False):
    """
    Translates the contents of an xarray.DataArray into a scipy.interpolate.RegularGridInterpolator.

    This is principally used for coordinate translations.
    """
    flip_axes = set()
    for d in arr.dims:
        c = arr.coords[d]
        if len(c) > 1 and c[1] - c[0] < 0:
            flip_axes.add(d)

    values = arr.values
    for dim in flip_axes:
        values = np.flip(values, arr.dims.index(dim))

    return scipy.interpolate.RegularGridInterpolator(
        points=[arr.coords[d].values[::-1] if d in flip_axes else arr.coords[d].values for d in arr.dims],
        values=values,
        bounds_error=bounds_error, fill_value=fill_value, method=method)


def convert_to_kspace(arr: xr.DataArray, resolution=None, **kwargs):
    old_dims = list(deepcopy(arr.dims))
    old_dims.remove('eV')
    old_dims.sort()

    if len(old_dims) == 0:
        # Was a core level scan or something similar
        return arr

    converted_dims = ['eV'] + {
        ('phi',): ['kp'],
        ('phi', 'polar'): ['kx', 'ky'],
        ('hv', 'phi'): ['kp', 'kz'],
        ('hv', 'phi', 'polar'): ['kx', 'ky', 'kz'],
    }.get(tuple(old_dims))

    convert_cls = {
        ('phi',): ConvertKp,
        ('phi', 'polar'): ConvertKxKy,
        ('hv', 'phi'): ConvertKpKz,
    }.get(tuple(old_dims))
    converter = convert_cls(arr)
    converted_coordinates = converter.get_coordinates(resolution)

    converted_arr = convert_coordinates(
        arr,
        converted_coordinates,
        {
            'dims': converted_dims,
            'transforms': dict(zip(arr.dims, [converter.conversion_for(d) for d in arr.dims])),
        }
    )

    del converted_arr.attrs['id']
    provenance(converted_arr, arr, {
        'what': 'Automatically k-space converted',
        'by': 'convert_to_kspace',
    })

    return converted_arr


def convert_coordinates(arr: xr.DataArray, target_coordinates, coordinate_transform):
    ordered_source_dimensions = arr.dims
    grid_interpolator = grid_interpolator_from_dataarray(
        arr.transpose(*ordered_source_dimensions), fill_value=float('nan'))

    # Skip the Jacobian correction for now
    # Convert the raw coordinate axes to a set of gridded points
    meshed_coordinates = np.meshgrid(*[target_coordinates[dim] for dim in target_coordinates], indexing='ij')
    meshed_coordinates = [meshed_coord.ravel() for meshed_coord in meshed_coordinates]

    ordered_transformations = [coordinate_transform['transforms'][dim] for dim in arr.dims]
    converted_volume = grid_interpolator(np.array([tr(*meshed_coordinates) for tr in ordered_transformations]).T)

    # Wrap it all up
    return xr.DataArray(
        np.reshape(converted_volume, [len(target_coordinates[d]) for d in target_coordinates], order='C'),
        target_coordinates,
        target_coordinates,
        attrs=arr.attrs
    )


def ke_polar_to_kx(kinetic_energy, theta, phi, metadata: dict = None):
    """
    Convert from standard polar angles p_theta, p_phi to the k-space Cartesian
    coordinate p_x.

    This function does not use its metadata argument, but retains it for symmetry
    with the other methods.
    """
    return (consts.K_INV_ANGSTROM * np.sqrt(kinetic_energy) *
            np.sin(theta) * np.cos(phi))


def ke_polar_to_kz(kinetic_energy, theta, phi, metadata: dict = None):
    """
    Convert from standard polar angles p_theta, p_phi to the k-space Cartesian
    coordinate p_z.

    This function requires knowing what the inner potential is in order to calculate
    p_z, so metadata has to contain a key-value pair for this constant.
    """
    return (consts.K_INV_ANGSTROM * np.sqrt(
        kinetic_energy * np.cos(theta) ** 2) + metadata['inner_potential'])


def kp_to_polar(energies, kp, metadata=None):
    return np.arcsin(kp / (consts.K_INV_ANGSTROM * np.sqrt(energies)))


def kx_kz_E_to_polar(kx, kz, E, metadata=None):
    k2 = kx ** 2 + kz ** 2
    return kp_to_polar(arpes.constants.HV_CONVERSION * k2 - metadata['inner_potential'] - E, kx)


def kx_kz_E_to_hv(kx, kz, E, metadata=None):
    k2 = kx ** 2 + kz ** 2
    return arpes.constants.HV_CONVERSION * k2 - metadata['inner_potential'] - E + metadata['work_function']


def polar_hv_E_to_kx(polar, hv, E, metadata=None):
    raise UnimplementedException('polar_hv_E_to_kx not implemented')


def polar_hv_E_to_kz(polar, hv, E, metadata=None):
    raise UnimplementedException('polar_hv_E_to_kz not implemented')


def polar_elev_KE_to_kx(polar, elev, KE, metadata=None):
    return consts.K_INV_ANGSTROM * np.sqrt(KE) * np.sin(polar)


def polar_elev_KE_to_ky(polar, elev, KE, metadata=None):
    return consts.K_INV_ANGSTROM * np.sqrt(KE) * np.sin(elev)


def kx_ky_KE_to_polar(kx, ky, KE, metadata=None):
    return np.arcsin(kx / (consts.K_INV_ANGSTROM * np.sqrt(KE)) /
                     np.cos(kx_ky_KE_to_elev(kx, ky, KE, metadata)))


def kx_ky_KE_to_elev(kx, ky, KE, metadata=None):
    """
    Assumes no elevation offsets, this greatly simplifies the conversion, but you can
    refer to Kspace_JD.ipf for the full version
    """
    return np.arcsin(ky / (consts.K_INV_ANGSTROM * np.sqrt(KE)))


# Jacobians organized here
def jacobian_polar_hv_E_to_kx_kz_E(kx, kz, E, metadata=None):
    raise UnimplementedException('jacobian_polar_hv_E_to_kx_kz_E not implemented')


def jacobian_polar_elev_KE_to_kx_ky_KE(kx, ky, E, metadata=None):
    raise UnimplementedException('jacobian_polar_elev_E_to_kx_ky_E not implemented')


# Bounds finding methods here, this may work out to be inferrable from the rest
# of the translation functions, so it might be that these methods disappear
# as Conrad's code slowly improves
def polar_hv_E_corners_to_kx_kz_E_bounds(corners, metadata=None):
    polar_angles = [c[0] for c in corners]
    hvs = [c[1] for c in corners]
    BEs = [c[2] for c in corners]

    min_angle, max_angle = min(polar_angles), max(polar_angles)
    min_hv, max_hv = min(hvs), max(hvs)
    min_BE, max_BE = min(BEs), max(BEs)
    WF = metadata['work_function']

    kx_min = min([ke_polar_to_kx(max_hv - max_BE - WF, min_angle, 0, metadata=metadata),
                  ke_polar_to_kx(min_hv - max_BE - WF, min_angle, 0, metadata=metadata)])
    kx_max = min([ke_polar_to_kx(max_hv - max_BE - WF, max_angle, 0, metadata=metadata),
                  ke_polar_to_kx(min_hv - max_BE - WF, max_angle, 0, metadata=metadata)])

    theta_max = max(abs(min_angle), abs(max_angle))
    kz_min = ke_polar_to_kz(min_hv + min_BE - WF, theta_max, 0, metadata=metadata)
    kz_max = ke_polar_to_kz(max_hv + max_BE - WF, 0, 0, metadata=metadata)

    return ((kx_min, kx_max,),
            (kz_min, kz_max,),
            (min_BE, max_BE,),)


def polar_elev_KE_corners_to_kx_ky_KE_bounds(corners, metadata=None):
    polars = [c[0] for c in corners]
    elevs = [c[1] for c in corners]
    KEs = [c[2] for c in corners]
    min_KE, max_KE = min(KEs), max(KEs)
    min_polar, max_polar = min(polars), max(polars)
    min_elev, max_elev = min(elevs), max(elevs)

    median_polar = (min_polar + max_polar) / 2
    median_elev = (min_elev + max_elev) / 2

    sampled_polars, sampled_elevs = np.array(
        list(itertools.product([min_polar, median_polar, max_polar],
                               [min_elev, median_elev, max_elev]))).T

    # this might be off by the material work function, but I could be wrong
    KE_at_eF = metadata['photon_energy'] - metadata['work_function']

    skxs = polar_elev_KE_to_kx(sampled_polars, sampled_elevs, KE_at_eF, metadata)
    skys = polar_elev_KE_to_ky(sampled_polars, sampled_elevs, KE_at_eF, metadata)

    return ((min(skxs), max(skxs)),
            (min(skys), max(skys)),
            (min_KE, max_KE))
