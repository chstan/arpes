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

Kinetic energy -> 'KE'
Photon energy -> 'hv'
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

HV_CONVERSION = 3.81

FINE_K_GRAINING = 0.01
MEDIUM_K_GRAINING = 0.05
COARSE_K_GRAINING = 0.1

# koutw = 0.5123*sqrt(hv-WF+BE)
#
# IF(geometry == 3) // Sample
#     # Polar - Tilt + Detector Vertical Slit(polw=detector angle)
#
#     t
#
#     newvol = interp3D(vol, polEw, tiltEw, z)
# ENDIF

class CoordinateConverter(object):
    def __init__(self, arr: xr.DataArray, *args, **kwargs):
        # Intern the volume so that we can check on things during computation
        self.arr = arr

    def prep(self, arr: xr.DataArray):
        """
        The CoordinateConverter.prep method allows you to precompute some transformations
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


class ConvertKxKy(CoordinateConverter):
    # TODO Convert all angles to radians where possible, avoid computations back and forth
    def __init__(self, *args, **kwargs):
        super(ConvertKxKy, self).__init__(*args, **kwargs)
        self.polar = None
        self.k_tot = None

    def kspace_to_BE(self, BE, *args, **kwargs):
        return BE

    def compute_k_tot(self, BE):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            self.arr.attrs['hv'] - self.arr.attrs.get('sample_workfunction', 4.32) + BE)

    #tiltEw = r2d * asin(ky / k_tot)

    #     polEw = r2d * asin(x / koutw(z) / cos(tiltEw / r2d))
    def kspace_to_phi(self, BE, kx, ky, *args, **kwargs):
        if self.k_tot is None:
           self.compute_k_tot(BE)

        if self.polar is None:
            self.kspace_to_polar(BE, kx, ky, *args, **kwargs)

        return (180 / np.pi) * np.arcsin(kx / self.k_tot / np.cos(self.polar * np.pi / 180))

    def kspace_to_polar(self, BE, kx, ky, *args, **kwargs):
        if self.k_tot is None:
            self.compute_k_tot(BE)

        self.polar = (180 / np.pi) * np.arcsin(ky / self.k_tot)
        return self.polar

def calculate_kx_ky_bounds(arr: xr.DataArray):
    """
    Calculates the kx and ky range for a dataset with a fixed photon energy

    This is used to infer the gridding that should be used for a k-space conversion.
    Based on Jonathan Denlinger's old codes
    :param arr: Dataset that includes a key indicating the photon energy of the scan
    :return: ((kx_low, kx_high,), (ky_low, ky_high,))
    """
    phi_coords, polar_coords = arr.coords['phi'], arr.coords['polar']

    # Sample hopefully representatively along the edges
    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    polar_low, polar_high = np.min(polar_coords), np.max(polar_coords)
    phi_mid = (phi_high + phi_low) / 2
    polar_mid = (polar_high + polar_low) / 2

    sampled_phi_values = np.array([phi_high, phi_high, phi_mid, phi_low, phi_low,
                                   phi_low, phi_mid, phi_high, phi_high])
    sampled_polar_values = np.array([polar_mid, polar_high, polar_high, polar_high, polar_mid,
                                     polar_low, polar_low, polar_low, polar_mid])
    KE = arr.attrs['hv'] - arr.attrs.get('sample_work_function', 4.32)

    kxs = arpes.constants.K_INV_ANGSTROM * np.sqrt(KE) * \
                        np.sin(np.pi / 180. * sampled_phi_values)
    kys = arpes.constants.K_INV_ANGSTROM * np.sqrt(KE) * \
                        np.cos(np.pi / 180. * sampled_phi_values) * \
                        np.sin(np.pi / 180. * sampled_polar_values)

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
    assert('eV' in old_coords)
    old_coords.remove('eV')
    old_coords.sort()

    new_coords = {
        ('phi',): ['kp'],
        ('phi','polar',): ['kx', 'ky'],
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
                                     bounds_error=False, **kwargs):
    """
    Translates the contents of an xarray.DataArray into a scipy.interpolate.RegularGridInterpolator.

    This is principally used for coordinate translations.
    """
    return scipy.interpolate.RegularGridInterpolator(
        points=[arr.coords[d] for d in arr.dims], values=arr.values,
        bounds_error=bounds_error, fill_value=fill_value, method=method, **kwargs)


def convert_coordinates(arr: xr.DataArray, target_coordinates, coordinate_transform):
    target_dimensions = coordinate_transform['dims']
    transforms = coordinate_transform['transforms']

    ordered_source_dimensions = arr.dims
    grid_interpolator = grid_interpolator_from_dataarray(
        arr.transpose(*ordered_source_dimensions), fill_value=float('nan'))

    # Skip the Jacobian correction for now
    # Convert the raw coordinate axes to a set of gridded points
    meshed_coordinates = np.meshgrid(*[target_coordinates[dim] for dim in target_dimensions], indexing='ij')
    meshed_coordinates = [meshed_coord.ravel() for meshed_coord in meshed_coordinates]

    ordered_transformations = [coordinate_transform['transforms'][dim] for dim in arr.dims]
    converted_volume = grid_interpolator(np.array([tr(*meshed_coordinates) for tr in ordered_transformations]).T)

    # Wrap it all up
    return xr.DataArray(
        np.reshape(converted_volume, [len(target_coordinates[d]) for d in target_dimensions], order='C'),
        target_coordinates,
        target_dimensions,
        attrs=arr.attrs
    )

def ke_polar_to_kx(kinetic_energy, theta, phi, metadata: dict=None):
    """
    Convert from standard polar angles p_theta, p_phi to the k-space Cartesian
    coordinate p_x.

    This function does not use its metadata argument, but retains it for symmetry
    with the other methods.
    """
    return (consts.K_INV_ANGSTROM * np.sqrt(kinetic_energy) *
            np.sin(theta) * np.cos(phi))


def ke_polar_to_kz(kinetic_energy, theta, phi, metadata: dict=None):
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
    return kp_to_polar(HV_CONVERSION * k2 - metadata['inner_potential'] - E, kx)


def kx_kz_E_to_hv(kx, kz, E, metadata=None):
    k2 = kx ** 2 + kz ** 2
    return HV_CONVERSION * k2 - metadata['inner_potential'] - E + metadata['work_function']


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
