import numpy as np

import arpes.constants
from .base import *
from .bounds_calculations import *

__all__ = ['ConvertKp', 'ConvertKxKy']

class ConvertKp(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super(ConvertKp, self).__init__(*args, **kwargs)
        self.k_tot = None

    def get_coordinates(self, resolution: dict=None):
        if resolution is None:
            resolution = {}

        coordinates = super(ConvertKp, self).get_coordinates(resolution)
        (kp_low, kp_high) = calculate_kp_bounds(self.arr)

        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords['phi'])
        inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]

        coordinates['kp'] = np.arange(kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER,
                                      resolution.get('kp', inferred_kp_res))

        base_coords = {k: v for k, v in self.arr.coords.items()
                       if k not in ['eV', 'phi', 'polar']}

        coordinates.update(base_coords)
        return coordinates

    def compute_k_tot(self, binding_energy):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            self.arr.S.hv - self.arr.S.work_function + binding_energy)

    def kspace_to_phi(self, binding_energy, kp, *args, **kwargs):
        polar_angle = self.arr.S.polar - self.arr.S.polar_offset

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        # TODO verify this
        return np.arcsin(kp / self.k_tot / np.cos(polar_angle)) + self.arr.S.phi_offset


    def conversion_for(self, dim):
        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            'eV': self.kspace_to_BE,
            'phi': self.kspace_to_phi
        }.get(dim, with_identity)


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

        inferred_kx_res = (kx_high - kx_low + 2 * K_SPACE_BORDER) / len(self.arr.coords['phi'])
        inferred_kx_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kx_res][-1]

        inferred_ky_res = (ky_high - ky_low + 2 * K_SPACE_BORDER) / len(self.arr.coords['polar'])
        if len(self.arr.coords['polar']) < 80: # This is arbitrary
            # Go a little finer here, so that we are not undersampling along the perpendicular directions
            # for dispersive features
            inferred_ky_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_ky_res][-2]
        else:
            # We should be well enough sampled
            inferred_ky_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_ky_res][-1]

        coordinates['kx'] = np.arange(kx_low - K_SPACE_BORDER, kx_high + K_SPACE_BORDER,
                                      resolution.get('kx', inferred_kx_res))
        coordinates['ky'] = np.arange(ky_low - K_SPACE_BORDER, ky_high + K_SPACE_BORDER,
                                      resolution.get('ky', inferred_ky_res))

        base_coords = {k: v for k, v in self.arr.coords.items()
                       if k not in ['eV', 'phi', 'polar']}
        coordinates.update(base_coords)

        return coordinates

    def compute_k_tot(self, binding_energy):
        self.k_tot = arpes.constants.K_INV_ANGSTROM * np.sqrt(
            self.arr.S.hv - self.arr.S.work_function + binding_energy)

    def kspace_to_phi(self, binding_energy, kx, ky, *args, **kwargs):
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.polar is None:
            self.kspace_to_polar(binding_energy, kx, ky, *args, **kwargs)

        if self.is_slit_vertical:
            return np.arcsin(kx / self.k_tot / np.cos(self.polar - self.arr.S.polar_offset)) + \
                   self.arr.S.phi_offset
        else:
            return np.arcsin(kx / self.k_tot) + self.arr.S.phi_offset

    def kspace_to_polar(self, binding_energy, kx, ky, *args, **kwargs):
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.is_slit_vertical:
            self.polar = np.arcsin(ky / self.k_tot) + self.arr.S.polar_offset
        else:
            self.polar = np.arcsin(ky / np.sqrt(self.k_tot ** 2 - kx ** 2)) + self.arr.S.polar_offset
        return self.polar

    def conversion_for(self, dim):
        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            'eV': self.kspace_to_BE,
            'phi': self.kspace_to_phi,
            'polar': self.kspace_to_polar,
        }.get(dim, with_identity)


