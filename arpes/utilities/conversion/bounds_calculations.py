import numpy as np
import xarray as xr

import arpes.constants

__all__ = ('calculate_kp_kz_bounds', 'calculate_kx_ky_bounds', 'calculate_kp_bounds')


def euler_to_kx(kinetic_energy, phi, polar, theta=0, slit_is_vertical=False):
    if slit_is_vertical:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(polar) * np.cos(phi)
    else:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(phi + theta)


def euler_to_ky(kinetic_energy, phi, polar, theta=0, slit_is_vertical=False):
    if slit_is_vertical:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * (
            np.cos(theta) * np.sin(phi) + np.cos(polar) * np.cos(phi) * np.sin(theta)
        )
    else:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * (
            np.cos(phi + theta) * np.sin(polar),
        )

def euler_to_kz(kinetic_energy, phi, polar, theta=0, inner_potential=10, slit_is_vertical=False):
    if slit_is_vertical:
        polar_term = -np.sin(theta) * np.sin(phi) + np.cos(theta) * np.cos(polar) * np.cos(phi)

    else:
        polar_term = np.cos(phi + theta) * np.cos(polar)

    return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy * polar_term ** 2 + inner_potential)

def spherical_to_kx(kinetic_energy, theta, phi):
    return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(theta) * np.cos(phi)

def spherical_to_ky(kinetic_energy, theta, phi):
    return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(theta) * np.sin(phi)

def spherical_to_kz(kinetic_energy, theta, phi, inner_V):
    """
    K_INV_ANGSTROM encodes that k_z = \frac{\sqrt{2 * m * E_kin * \cos^2\theta + V_0}}{\hbar}
    :param kinetic_energy:
    :param theta:
    :param phi:
    :param inner_V:
    :return:
    """
    return arpes.constants.K_INV_ANGSTROM * np.sqrt(
        kinetic_energy * np.cos(theta) ** 2 + inner_V)


def calculate_kp_kz_bounds(arr: xr.DataArray):
    phi_offset = arr.S.phi_offset
    phi_min = np.min(arr.coords['phi'].values) - phi_offset
    phi_max = np.max(arr.coords['phi'].values) - phi_offset

    binding_energy_min, binding_energy_max = np.min(arr.coords['eV'].values), np.max(arr.coords['eV'].values)
    hv_min, hv_max = np.min(arr.coords['hv'].values), np.max(arr.coords['hv'].values)

    wf = arr.S.work_function
    kx_min = min(spherical_to_kx(hv_max - binding_energy_max - wf, phi_min, 0),
                 spherical_to_kx(hv_min - binding_energy_max - wf, phi_min, 0))
    kx_max = max(spherical_to_kx(hv_max - binding_energy_max - wf, phi_max, 0),
                 spherical_to_kx(hv_min - binding_energy_max - wf, phi_max, 0))

    angle_max = max(abs(phi_min), abs(phi_max))
    inner_V = arr.S.inner_potential
    kz_min = spherical_to_kz(hv_min + binding_energy_min - wf, angle_max, 0, inner_V)
    kz_max = spherical_to_kz(hv_max + binding_energy_max - wf, 0, 0, inner_V)

    return (
        (round(kx_min, 2), round(kx_max, 2)), # kp
        (round(kz_min, 2), round(kz_max, 2)), # kz
    )


def calculate_kp_bounds(arr: xr.DataArray):
    phi_coords = arr.coords['phi'].values - arr.S.phi_offset
    polar = float(arr.attrs.get('polar')) - arr.S.polar_offset

    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    phi_mid = (phi_high + phi_low) / 2

    sampled_phi_values = np.array([phi_low, phi_mid, phi_high])

    kinetic_energy = arr.S.hv - arr.S.work_function
    kps = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(sampled_phi_values) * np.cos(polar)

    return round(np.min(kps), 2), round(np.max(kps), 2)


def calculate_kx_ky_bounds(arr: xr.DataArray):
    """
    Calculates the kx and ky range for a dataset with a fixed photon energy

    This is used to infer the gridding that should be used for a k-space conversion.
    Based on Jonathan Denlinger's old codes
    :param arr: Dataset that includes a key indicating the photon energy of the scan
    :return: ((kx_low, kx_high,), (ky_low, ky_high,))
    """
    phi_coords, polar_coords = arr.coords['phi'] - arr.S.phi_offset, arr.coords['polar'] - arr.S.polar_offset

    # Sample hopefully representatively along the edges
    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    polar_low, polar_high = np.min(polar_coords), np.max(polar_coords)
    phi_mid = (phi_high + phi_low) / 2
    polar_mid = (polar_high + polar_low) / 2

    sampled_phi_values = np.array([phi_high, phi_high, phi_mid, phi_low, phi_low,
                                   phi_low, phi_mid, phi_high, phi_high])
    sampled_polar_values = np.array([polar_mid, polar_high, polar_high, polar_high, polar_mid,
                                     polar_low, polar_low, polar_low, polar_mid])
    kinetic_energy = arr.S.hv - arr.S.work_function

    kxs = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(sampled_phi_values)
    kys = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.cos(sampled_phi_values) * np.sin(sampled_polar_values)

    return (
        (round(np.min(kxs), 2), round(np.max(kxs), 2)),
        (round(np.min(kys), 2), round(np.max(kys), 2)),
    )


