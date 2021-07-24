"""Coordinate conversion classes for photon energy scans."""
import numpy as np
import numba

import arpes.constants
from typing import Any, Callable, Dict

from .base import CoordinateConverter, K_SPACE_BORDER, MOMENTUM_BREAKPOINTS
from .bounds_calculations import calculate_kp_kz_bounds

__all__ = ["ConvertKpKzV0", "ConvertKxKyKz", "ConvertKpKz"]


@numba.njit(parallel=True, cache=True)
def _kspace_to_hv(kp, kz, hv, energy_shift, is_constant_shift):
    """Efficiently perform the inverse coordinate transform to photon energy."""
    shift_ratio = 0 if is_constant_shift else 1

    for i in numba.prange(len(kp)):
        hv[i] = (
            arpes.constants.HV_CONVERSION * (kp[i] ** 2 + kz[i] ** 2)
            + energy_shift[i * shift_ratio]
        )


@numba.njit(parallel=True, cache=True)
def _kp_to_polar(kinetic_energy, kp, phi, inner_potential, angle_offset):
    """Efficiently performs the inverse coordinate transform phi(hv, kp)."""
    for i in numba.prange(len(kp)):
        phi[i] = (
            np.arcsin(
                kp[i]
                / (arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy[i] + inner_potential))
            )
            + angle_offset
        )


class ConvertKpKzV0(CoordinateConverter):
    """Implements inner potential broadcasted hv Fermi surfaces."""

    # TODO implement
    def __init__(self, *args, **kwargs):
        """TODO, implement this."""
        super(ConvertKpKzV0, self).__init__(*args, **kwargs)
        raise NotImplementedError


class ConvertKxKyKz(CoordinateConverter):
    """Implements 4D data volume conversion."""

    def __init__(self, *args, **kwargs):
        """TODO, implement this."""
        super(ConvertKxKyKz, self).__init__(*args, **kwargs)
        raise NotImplementedError


class ConvertKpKz(CoordinateConverter):
    """Implements single angle photon energy scans."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Cache the photon energy coordinate we calculate backwards from kz."""
        super(ConvertKpKz, self).__init__(*args, **kwargs)
        self.hv = None
        self.phi = None

    def get_coordinates(
        self, resolution: dict = None, bounds: dict = None
    ) -> Dict[str, np.ndarray]:
        """Calculates appropriate coordinate bounds."""
        if resolution is None:
            resolution = {}
        if bounds is None:
            bounds = {}

        coordinates = super(ConvertKpKz, self).get_coordinates(resolution=resolution, bounds=bounds)

        ((kp_low, kp_high), (kz_low, kz_high)) = calculate_kp_kz_bounds(self.arr)
        if "kp" in bounds:
            kp_low, kp_high = bounds["kp"]

        if "kz" in bounds:
            kz_low, kz_high = bounds["kz"]

        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["phi"])
        inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]

        # go a bit finer here because it would otherwise be very coarse
        inferred_kz_res = (kz_high - kz_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["hv"])
        inferred_kz_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kz_res][-1]

        coordinates["kp"] = np.arange(
            kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER, resolution.get("kp", inferred_kp_res)
        )
        coordinates["kz"] = np.arange(
            kz_low - K_SPACE_BORDER, kz_high + K_SPACE_BORDER, resolution.get("kz", inferred_kz_res)
        )

        base_coords = {k: v for k, v in self.arr.coords.items() if k not in ["eV", "phi", "hv"]}

        coordinates.update(base_coords)

        return coordinates

    def kspace_to_hv(
        self, binding_energy: np.ndarray, kp: np.ndarray, kz: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the raw photon energy."""
        if self.hv is None:
            inner_v = self.arr.S.inner_potential
            wf = self.arr.S.work_function

            is_constant_shift = True
            if not isinstance(binding_energy, np.ndarray):
                is_constant_shift = True
                binding_energy = np.array([binding_energy])

            self.hv = np.zeros_like(kp)
            _kspace_to_hv(kp, kz, self.hv, -inner_v - binding_energy + wf, is_constant_shift)

        return self.hv

    def kspace_to_phi(
        self, binding_energy: np.ndarray, kp: np.ndarray, kz: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the hemisphere angle axis."""
        if self.phi is not None:
            return self.phi

        if self.hv is None:
            self.kspace_to_hv(binding_energy, kp, kz, *args, **kwargs)

        kinetic_energy = binding_energy + self.hv - self.arr.S.work_function

        self.phi = np.zeros_like(self.hv)

        _kp_to_polar(
            kinetic_energy,
            kp,
            self.phi,
            self.arr.S.inner_potential,
            self.arr.S.phi_offset,
        )

        try:
            self.phi = self.calibration.correct_detector_angle(eV=binding_energy, phi=self.phi)
        except:
            pass

        return self.phi

    def conversion_for(self, dim: str) -> Callable:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            "eV": self.kspace_to_BE,
            "hv": self.kspace_to_hv,
            "phi": self.kspace_to_phi,
        }.get(dim, with_identity)
