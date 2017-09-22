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
"""

# pylint: disable=W0613, C0103

import itertools

import numpy

import arpes.constants as consts
from arpes.exceptions import UnimplementedException

HV_CONVERSION = 3.81


def ke_polar_to_kx(kinetic_energy, theta, phi, metadata: dict=None):
    """
    Convert from standard polar angles p_theta, p_phi to the k-space Cartesian
    coordinate p_x.

    This function does not use its metadata argument, but retains it for symmetry
    with the other methods.
    """
    return (consts.K_INV_ANGSTROM * numpy.sqrt(kinetic_energy) *
            numpy.sin(theta) * numpy.cos(phi))


def ke_polar_to_kz(kinetic_energy, theta, phi, metadata: dict=None):
    """
    Convert from standard polar angles p_theta, p_phi to the k-space Cartesian
    coordinate p_z.

    This function requires knowing what the inner potential is in order to calculate
    p_z, so metadata has to contain a key-value pair for this constant.
    """
    return (consts.K_INV_ANGSTROM * numpy.sqrt(
        kinetic_energy * numpy.cos(theta) ** 2) + metadata['inner_potential'])


def kp_to_polar(energies, kp, metadata=None):
    return numpy.arcsin(kp / (consts.K_INV_ANGSTROM * numpy.sqrt(energies)))


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
    return consts.K_INV_ANGSTROM * numpy.sqrt(KE) * numpy.sin(polar)


def polar_elev_KE_to_ky(polar, elev, KE, metadata=None):
    return consts.K_INV_ANGSTROM * numpy.sqrt(KE) * numpy.sin(elev)


def kx_ky_KE_to_polar(kx, ky, KE, metadata=None):
    return numpy.arcsin(kx / (consts.K_INV_ANGSTROM * numpy.sqrt(KE)) /
                        numpy.cos(kx_ky_KE_to_elev(kx, ky, KE, metadata)))


def kx_ky_KE_to_elev(kx, ky, KE, metadata=None):
    """
    Assumes no elevation offsets, this greatly simplifies the conversion, but you can
    refer to Kspace_JD.ipf for the full version
    """
    return numpy.arcsin(ky / (consts.K_INV_ANGSTROM * numpy.sqrt(E)))


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

    sampled_polars, sampled_elevs = numpy.array(
        list(itertools.product([min_polar, median_polar, max_polar],
                               [min_elev, median_elev, max_elev]))).T

    # this might be off by the material work function, but I could be wrong
    KE_at_eF = metadata['photon_energy'] - metadata['work_function']

    skxs = polar_elev_KE_to_kx(sampled_polars, sampled_elevs, KE_at_eF, metadata)
    skys = polar_elev_KE_to_ky(sampled_polars, sampled_elevs, KE_at_eF, metadata)

    return ((min(skxs), max(skxs)),
            (min(skys), max(skys)),
            (min_KE, max_KE))
