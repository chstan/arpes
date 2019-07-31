import numpy as np

import arpes.constants
from .base import *
from .bounds_calculations import *

__all__ = ['ConvertKp', 'ConvertKxKy']

class ConvertKp(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_tot = None

    def get_coordinates(self, resolution: dict=None):
        if resolution is None:
            resolution = {}

        coordinates = super().get_coordinates(resolution)
        (kp_low, kp_high) = calculate_kp_bounds(self.arr)

        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords['phi'])
        inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]

        coordinates['kp'] = np.arange(kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER,
                                      resolution.get('kp', inferred_kp_res))

        base_coords = {k: v for k, v in self.arr.coords.items()
                       if k not in ['eV', 'phi', 'beta', 'theta']}

        coordinates.update(base_coords)
        return coordinates

    def compute_k_tot(self, binding_energy):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            self.arr.S.hv - self.arr.S.work_function + binding_energy)

    def kspace_to_phi(self, binding_energy, kp, *args, **kwargs):
        # check signs again here
        if self.is_slit_vertical:
            polar_angle = self.arr.S.lookup_offset_coord('theta') + self.arr.S.lookup_offset_coord('psi')
            parallel_angle = self.arr.S.lookup_offset_coord('beta')
        else:
            polar_angle = self.arr.S.lookup_offset_coord('beta') + self.arr.S.lookup_offset_coord('psi')
            parallel_angle = self.arr.S.lookup_offset_coord('theta')

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        # TODO verify this
        return np.arcsin(kp / self.k_tot / np.cos(polar_angle)) + self.arr.S.phi_offset + parallel_angle

    def conversion_for(self, dim):
        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            'eV': self.kspace_to_BE,
            'phi': self.kspace_to_phi
        }.get(dim, with_identity)


class ConvertKxKy(CoordinateConverter):
    """
    Please note that currently we assume that psi = 0 when you are not using an
    electrostatic deflector
    """
    def __init__(self, arr, *args, **kwargs):
        super().__init__(arr, *args, **kwargs)
        self.k_tot = None
        self.phi = None
        # the angle perpendicular to phi as appropriate to the scan, this can be any of
        # psi, theta, beta
        self.perp_angle = None

        self.rkx = None
        self.rky = None

        # accept either vertical or horizontal, fail otherwise
        if not any(np.abs(arr.alpha - alpha_option) < (np.pi / 180) for alpha_option in [0, np.pi/2]):
            raise ValueError('You must convert either vertical or horizontal slit data with this converter.')

        self.direct_angles = ('phi', [d for d in ['psi', 'beta', 'theta'] if d in arr.indexes][0])

        if self.direct_angles[1] != 'psi':
            # psi allows for either orientation
            assert (self.direct_angles[1] in {'theta'}) != (not self.is_slit_vertical)

        # determine which other angles constitute equivalent sets
        if self.is_slit_vertical:
            self.parallel_angles = ('beta', 'theta')
        else:
            self.parallel_angles = ('theta', 'beta')

    def get_coordinates(self, resolution: dict = None):
        if resolution is None:
            resolution = {}

        coordinates = super().get_coordinates(resolution)

        ((kx_low, kx_high), (ky_low, ky_high)) = calculate_kx_ky_bounds(self.arr)

        kx_angle, ky_angle = self.direct_angles
        if self.is_slit_vertical:
            # phi actually measures along ky
            ky_angle, kx_angle = kx_angle, ky_angle

        len_ky_angle = len(self.arr.coords[ky_angle])
        len_kx_angle = len(self.arr.coords[kx_angle])

        inferred_kx_res = (kx_high - kx_low + 2 * K_SPACE_BORDER) / len(self.arr.coords[kx_angle])
        inferred_ky_res = (ky_high - ky_low + 2 * K_SPACE_BORDER) / len(self.arr.coords[ky_angle])

        # upsample a bit if there aren't that many points along a certain axis
        try:
            inferred_kx_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kx_res][-2 if (len_kx_angle < 80) else -1]
        except IndexError:
            inferred_kx_res = MOMENTUM_BREAKPOINTS[-2]
        try:
            inferred_ky_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_ky_res][-2 if (len_ky_angle < 80) else -1]
        except IndexError:
            inferred_ky_res = MOMENTUM_BREAKPOINTS[-2]

        coordinates['kx'] = np.arange(kx_low - K_SPACE_BORDER, kx_high + K_SPACE_BORDER,
                                      resolution.get('kx', inferred_kx_res))
        coordinates['ky'] = np.arange(ky_low - K_SPACE_BORDER, ky_high + K_SPACE_BORDER,
                                      resolution.get('ky', inferred_ky_res))

        base_coords = {k: v for k, v in self.arr.coords.items()
                       if k not in ['eV', 'phi', 'psi', 'theta', 'beta', 'alpha', 'chi']}
        coordinates.update(base_coords)

        return coordinates

    def compute_k_tot(self, binding_energy):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            self.arr.S.hv - self.arr.S.work_function + binding_energy)

    def conversion_for(self, dim):
        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            'eV': self.kspace_to_BE,
            'phi': self.kspace_to_phi,
            'theta': self.kspace_to_perp_angle,
            'psi': self.kspace_to_perp_angle,
            'beta': self.kspace_to_perp_angle,
        }.get(dim, with_identity)

    @property
    def needs_rotation(self):
        # force rotation when greater than 0.5 deg
        return np.abs(self.arr.S.lookup_offset_coord('chi')) > (0.5 * np.pi / 180)

    def rkx_rky(self, kx, ky):
        """
        Returns the rotated kx and ky values when we are rotating by nonzero chi
        :return:
        """

        if self.rkx is not None:
            return self.rkx, self.rky

        chi = self.arr.S.lookup_offset_coord('chi')

        self.rkx = kx * np.cos(chi) - ky * np.sin(chi)
        self.rky = ky * np.cos(chi) + kx * np.sin(chi)

        return self.rkx, self.rky

    def kspace_to_phi(self, binding_energy, kx, ky, *args, **kwargs):
        if self.phi is not None:
            return self.phi

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)

        # This can be condensed but it is actually better not to condense it:
        # In this format, we can very easily compare to the raw coordinate conversion functions that
        # come from Mathematica in order to adjust signs, etc.
        scan_angle = self.direct_angles[1]
        if scan_angle == 'psi':
            if self.is_slit_vertical:
                self.phi = np.arcsin(ky / np.sqrt(self.k_tot ** 2 - kx ** 2)) + self.arr.S.phi_offset + \
                           self.arr.S.lookup_offset_coord(self.parallel_angles[0])
            else:
                self.phi = np.arcsin(kx / np.sqrt(self.k_tot ** 2 - ky ** 2)) + self.arr.S.phi_offset + \
                           self.arr.S.lookup_offset_coord(self.parallel_angles[0])
        elif scan_angle == 'beta':
            # vertical slit
            self.phi = np.arcsin(kx / self.k_tot) + self.arr.S.phi_offset + \
                       self.arr.S.lookup_offset_coord(self.parallel_angles[0])
        elif scan_angle == 'theta':
            # vertical slit
            self.phi = np.arcsin(ky / self.k_tot) + self.arr.S.phi_offset + \
                       self.arr.S.lookup_offset_coord(self.parallel_angles[0])
        else:
            raise ValueError('No recognized scan angle found for {}'.format(self.parallel_angles[1]))

        return self.phi

    def kspace_to_perp_angle(self, binding_energy, kx, ky, *args, **kwargs):
        if self.perp_angle is not None:
            return self.perp_angle

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)

        scan_angle = self.direct_angles[1]
        if scan_angle == 'psi':
            if self.is_slit_vertical:
                self.perp_angle = -np.arcsin(kx / self.k_tot) + self.arr.S.psi_offset - \
                                  self.arr.S.lookup_offset_coord(self.parallel_angles[1])
            else:
                self.perp_angle = np.arcsin(ky / self.k_tot) + self.arr.S.psi_offset + \
                                  self.arr.S.lookup_offset_coord(self.parallel_angles[1])
        elif scan_angle == 'beta':
            self.perp_angle = -np.arcsin(ky / np.sqrt(self.k_tot ** 2 - kx ** 2)) + self.arr.S.beta_offset
            pass
        elif scan_angle == 'theta':
            self.perp_angle = -np.arcsin(kx / np.sqrt(self.k_tot ** 2 - ky ** 2)) + self.arr.S.theta_offset
            pass
        else:
            raise ValueError('No recognized scan angle found for {}'.format(self.parallel_angles[1]))

        return self.perp_angle