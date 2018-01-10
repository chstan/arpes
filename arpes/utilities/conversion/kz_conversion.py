import numpy as np

import arpes.constants
from .base import *
from .bounds_calculations import *

__all__ = ['ConvertKpKzV0', 'ConvertKxKyKz', 'ConvertKpKz']

class ConvertKpKzV0(CoordinateConverter):
    # TODO implement
    def __init__(self, *args, **kwargs):
        super(ConvertKpKzV0, self).__init__(*args, **kwargs)
        raise NotImplementedError()


class ConvertKxKyKz(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super(ConvertKxKyKz, self).__init__(*args, **kwargs)


class ConvertKpKz(CoordinateConverter):
    def __init__(self, *args, **kwargs):
        super(ConvertKpKz, self).__init__(*args, **kwargs)
        self.hv = None

    def get_coordinates(self, resolution: dict=None):
        if resolution is None:
            resolution = {}

        coordinates = super(ConvertKpKz, self).get_coordinates(resolution)

        ((kp_low, kp_high), (kz_low, kz_high)) = calculate_kp_kz_bounds(self.arr)

        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords['phi'])
        inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]

        # go a bit finer here because it would otherwise be very coarse
        inferred_kz_res = (kz_high - kz_low + 2 * K_SPACE_BORDER) / len(self.arr.coords['hv'])
        inferred_kz_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kz_res][-1]

        coordinates['kp'] = np.arange(kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER,
                                      resolution.get('kp', inferred_kp_res))
        coordinates['kz'] = np.arange(kz_low - K_SPACE_BORDER, kz_high + K_SPACE_BORDER,
                                      resolution.get('kz', inferred_kz_res))

        base_coords = {k: v for k, v in self.arr.coords.items()
                       if k not in ['eV', 'phi', 'hv']}

        coordinates.update(base_coords)

        return coordinates

    def kspace_to_hv(self, binding_energy, kp, kz, *args, **kwargs):
        # x = kp, y = kz, z = BE
        if self.hv is None:
            inner_v = self.arr.S.inner_potential
            wf = self.arr.S.work_function
            self.hv = arpes.constants.HV_CONVERSION * (kp ** 2 + kz ** 2) + (
                -inner_v - binding_energy + wf)

        return self.hv

    def kspace_to_phi(self, binding_energy, kp, kz, *args, **kwargs):
        if self.hv is None:
            self.kspace_to_hv(binding_energy, kp, kz, *args, **kwargs)

        def kp_to_polar(kinetic_energy_out, kp):
            return np.arcsin(kp / (arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy_out)))

        return kp_to_polar(self.hv + self.arr.S.work_function, kp) + self.arr.S.phi_offset

    def conversion_for(self, dim):
        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            'eV': self.kspace_to_BE,
            'hv': self.kspace_to_hv,
            'phi': self.kspace_to_phi,
        }.get(dim, with_identity)


