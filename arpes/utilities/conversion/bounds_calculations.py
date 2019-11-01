import numpy as np

import arpes.constants
import xarray as xr

__all__ = ('calculate_kp_kz_bounds', 'calculate_kx_ky_bounds', 'calculate_kp_bounds',
           'full_angles_to_k', 'full_angles_to_k_approx',)


def full_angles_to_k_approx(kinetic_energy, phi, psi, alpha, beta, theta, chi, inner_potential):
    """
    Small angle approximation of the momentum conversion functions. Depending on the value of alpha,
    which we do not small angle approximate, this takes a few different forms.

    :param kinetic_energy:
    :param phi:
    :param psi:
    :param alpha:
    :param beta:
    :param theta:
    :param chi:
    :param inner_potential:
    :return:
    """
    raise NotImplementedError()


def full_angles_to_k(kinetic_energy, phi, psi, alpha, beta, theta, chi, inner_potential, approximate=False):
    """
    Converts from the full set of standard PyARPES angles to momentum. More details on angle to momentum conversion
    can be found at `the momentum conversion notes <https://arpes.netlify.com/#/momentum-conversion>`_.

    Because the inverse coordinate transforms in PyARPES use the small angle approximation, we also allow
    the small angle approximation in the forward direction, using the `approximate=` keyword argument.
    :param kinetic_energy:
    :param phi:
    :param psi:
    :param alpha:
    :param beta:
    :param theta:
    :param chi:
    :param inner_potential:
    :param approximate:
    :return:
    """
    if approximate:
        return full_angles_to_k_approx(kinetic_energy, phi, psi, alpha, beta, theta, chi, inner_potential)

    theta, beta, chi, psi = theta, beta, -chi, psi

    # use the full direct momentum conversion
    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
    sin_beta, cos_beta = np.sin(beta), np.cos(beta)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_chi, cos_chi = np.sin(chi), np.cos(chi)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)

    vx = cos_alpha * cos_psi * sin_phi - sin_alpha * sin_psi
    vy = sin_alpha * cos_psi * sin_phi + cos_alpha * sin_psi
    vz = cos_phi * cos_psi

    # perform theta rotation
    vrtheta_x = cos_theta * vx - sin_theta * vz
    vrtheta_y = vy
    vrtheta_z = sin_theta * vx + cos_theta * vz

    # perform beta rotation
    vrbeta_x = vrtheta_x
    vrbeta_y = cos_beta * vrtheta_y - sin_beta * vrtheta_z
    vrbeta_z = sin_beta * vrtheta_y + cos_beta * vrtheta_z

    # Perform chi rotation
    vrchi_x = cos_chi * vrbeta_x - sin_chi * vrbeta_y
    vrchi_y = sin_chi * vrbeta_x + cos_chi * vrbeta_y
    vrchi_z = vrbeta_z

    v_par_sq = vrchi_x ** 2 + vrchi_y ** 2

    """
    velocity -> momentum in each of parallel and perpendicular directions
    in the perpendicular case, we need the value of the cos^2(zeta) for the polar declination
    angle zeta in the sample (emission) frame. The total in plane velocity v_parallel is proportional
    to sin(zeta), so by the trig identity:
    
    1 = cos^2(zeta) + sin^2(zeta)
    
    we may substitute cos^2(zeta) for 1 - sin^2(zeta) which is 1 - (vrchi_x **2 + vrchi_y ** 2) above.  
    """
    k_par = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy)
    k_perp = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy * (1 - v_par_sq) + inner_potential)

    return k_par * vrchi_x, k_par * vrchi_y, k_perp * vrchi_z


def euler_to_kx(kinetic_energy, phi, beta, theta=0, slit_is_vertical=False):
    if slit_is_vertical:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(beta) * np.cos(phi)
    else:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(phi + theta)


def euler_to_ky(kinetic_energy, phi, beta, theta=0, slit_is_vertical=False):
    if slit_is_vertical:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * (
            np.cos(theta) * np.sin(phi) + np.cos(beta) * np.cos(phi) * np.sin(theta)
        )
    else:
        return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * (
            np.cos(phi + theta) * np.sin(beta),
        )


def euler_to_kz(kinetic_energy, phi, beta, theta=0, inner_potential=10, slit_is_vertical=False):
    if slit_is_vertical:
        beta_term = -np.sin(theta) * np.sin(phi) + np.cos(theta) * np.cos(beta) * np.cos(phi)

    else:
        beta_term = np.cos(phi + theta) * np.cos(beta)

    return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy * beta_term ** 2 + inner_potential)


def spherical_to_kx(kinetic_energy: np.float, theta: np.float, phi: np.float) -> np.float:
    return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(theta) * np.cos(phi)


def spherical_to_ky(kinetic_energy, theta, phi):
    return arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(theta) * np.sin(phi)


def spherical_to_kz(kinetic_energy: np.float, theta: np.float, phi: np.float, inner_V: np.float) -> np.float:
    r"""
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
    beta = float(arr.coords['beta']) - arr.S.beta_offset

    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    phi_mid = (phi_high + phi_low) / 2

    sampled_phi_values = np.array([phi_low, phi_mid, phi_high])

    kinetic_energy = arr.S.hv - arr.S.work_function
    kps = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(sampled_phi_values) * np.cos(beta)

    return round(np.min(kps), 2), round(np.max(kps), 2)


def calculate_kx_ky_bounds(arr: xr.DataArray):
    """
    Calculates the kx and ky range for a dataset with a fixed photon energy

    This is used to infer the gridding that should be used for a k-space conversion.
    Based on Jonathan Denlinger's old codes
    :param arr: Dataset that includes a key indicating the photon energy of the scan
    :return: ((kx_low, kx_high,), (ky_low, ky_high,))
    """
    phi_coords, beta_coords = arr.coords['phi'] - arr.S.phi_offset, arr.coords['beta'] - arr.S.beta_offset

    # Sample hopefully representatively along the edges
    phi_low, phi_high = np.min(phi_coords), np.max(phi_coords)
    beta_low, beta_high = np.min(beta_coords), np.max(beta_coords)
    phi_mid = (phi_high + phi_low) / 2
    beta_mid = (beta_high + beta_low) / 2

    sampled_phi_values = np.array([phi_high, phi_high, phi_mid, phi_low, phi_low,
                                   phi_low, phi_mid, phi_high, phi_high])
    sampled_beta_values = np.array([beta_mid, beta_high, beta_high, beta_high, beta_mid,
                                     beta_low, beta_low, beta_low, beta_mid])
    kinetic_energy = arr.S.hv - arr.S.work_function

    kxs = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(sampled_phi_values)
    kys = arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.cos(sampled_phi_values) * np.sin(sampled_beta_values)

    return (
        (round(np.min(kxs), 2), round(np.max(kxs), 2)),
        (round(np.min(kys), 2), round(np.max(kys), 2)),
    )
