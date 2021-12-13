"""Implements forward and reverse trapezoidal corrections."""
import warnings
import numpy as np
import xarray as xr

import numba

from typing import Any, Callable, Dict, List

from arpes.trace import Trace, traceable
from arpes.utilities import normalize_to_spectrum

from .base import CoordinateConverter
from .core import convert_coordinates

__all__ = ["apply_trapezoidal_correction"]


@numba.njit(parallel=True)
def _phi_to_phi(energy, phi, phi_out, l_fermi, l_volt, r_fermi, r_volt):
    """Performs reverse coordinate interpolation using four angular waypoints.

    Args:
        energy: The binding energy in the corrected coordinate space
        phi: The angle in the corrected coordinate space
        phi_out: The array to populate with the measured phi angles
        l_fermi: The measured phi coordinate of the left edge of the hemisphere's range
           at the Fermi level
        l_volt: The measured phi coordinate of the left edge of the hemisphere's range
           at a binding energy of 1 eV (eV = -1.0)
        r_fermi: The measured phi coordinate of the right edge of the hemisphere's range
           at the Fermi level
        r_volt: The measured phi coordinate of the right edge of the hemisphere's range
           at a binding energy of 1 eV (eV = -1.0)
    """
    for i in numba.prange(len(phi)):
        l = l_fermi - energy[i] * (l_volt - l_fermi)
        r = r_fermi - energy[i] * (r_volt - r_fermi)

        # These are the forward equations, we can just invert them below
        # c = (phi[i] - l) / (r - l)
        # phi_out[i] = l_fermi + c * (r_fermi - l_fermi)

        dac_da = (r - l) / (r_fermi - l_fermi)
        phi_out[i] = (phi[i] - l_fermi) * dac_da + l


@numba.njit(parallel=True)
def _phi_to_phi_forward(energy, phi, phi_out, l_fermi, l_volt, r_fermi, r_volt):
    """The inverse transform to ``_phi_to_phi``. See that function for details."""
    for i in numba.prange(len(phi)):
        l = l_fermi - energy[i] * (l_volt - l_fermi)
        r = r_fermi - energy[i] * (r_volt - r_fermi)

        # These are the forward equations
        c = (phi[i] - l) / (r - l)
        phi_out[i] = l_fermi + c * (r_fermi - l_fermi)


class ConvertTrapezoidalCorrection(CoordinateConverter):
    """A converter for applying the trapezoidal correction to ARPES data."""

    def __init__(self, *args: Any, corners: List[Dict[str, float]], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.phi = None

        # we normalize the corners so that they are equivalent to four corners at the Fermi level
        # and one volt below.
        c1, c2, c3, c4 = sorted(corners, key=lambda x: x["phi"])
        c1, c2 = sorted([c1, c2], key=lambda x: x["eV"])
        c3, c4 = sorted([c3, c4], key=lambda x: x["eV"])

        # now, corners are in
        # (c1, c2, c3, c4) = (LL, UL, LR, UR) order

        left_per_volt = (c1["phi"] - c2["phi"]) / (c1["eV"] - c2["eV"])
        left_phi_fermi = c2["phi"] - c2["eV"] * left_per_volt
        left_phi_one_volt = left_phi_fermi - left_per_volt

        right_per_volt = (c3["phi"] - c4["phi"]) / (c3["eV"] - c4["eV"])
        right_phi_fermi = c3["phi"] - c4["eV"] * right_per_volt
        right_phi_one_volt = right_phi_fermi - right_per_volt

        self.corner_angles = (
            left_phi_fermi,
            left_phi_one_volt,
            right_phi_fermi,
            right_phi_one_volt,
        )

    def get_coordinates(self, *args, **kwargs):
        return self.arr.indexes

    def conversion_for(self, dim: str) -> Callable:
        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            "phi": self.phi_to_phi,
        }.get(dim, with_identity)

    def phi_to_phi(self, binding_energy: np.ndarray, phi: np.ndarray, *args: Any, **kwargs: Any):
        if self.phi is not None:
            return self.phi
        self.phi = np.zeros_like(phi)
        _phi_to_phi(binding_energy, phi, self.phi, *self.corner_angles)
        return self.phi

    def phi_to_phi_forward(
        self, binding_energy: np.ndarray, phi: np.ndarray, *args: Any, **kwargs: Any
    ):
        phi_out = np.zeros_like(phi)
        _phi_to_phi_forward(binding_energy, phi, phi_out, *self.corner_angles)
        return phi_out


@traceable
def apply_trapezoidal_correction(
    data: xr.DataArray, corners: List[Dict[str, float]], trace: Trace = None
) -> xr.DataArray:
    """Applies the trapezoidal correction to data in angular units by linearly interpolating slices.

    Shares some code with standard coordinate conversion, i.e. to momentum, because you can think of
    this as performing a coordinate conversion between two angular coordinate sets, the measured angles
    and the true angles.

    Args:
        data: The xarray instances to perform correction on
        corners: These don't actually have to be corners, but are waypoints of the conversion. Use points near the Fermi
            level and near the bottom of the spectrum just at the edge of recorded angular region.
        trace: A trace instance which can be used to enable execution tracing and debugging. Pass ``True`` to enable.


    Returns:
        The corrected data.
    """
    trace("Normalizing to spectrum")

    if isinstance(data, dict):
        warnings.warn(
            "Treating dict-like data as an attempt to forward convert a single coordinate."
        )
        converter = ConvertTrapezoidalCorrection(None, [], corners=corners)
        result = dict(data)
        result["phi"] = converter.phi_to_phi_forward(
            np.array([data["eV"]]), np.array([data["phi"]])
        )[0]
        return result

    if isinstance(data, xr.Dataset):
        warnings.warn(
            "Remember to use a DataArray not a Dataset, attempting to extract spectrum and copy attributes."
        )
        attrs = data.attrs.copy()
        data = normalize_to_spectrum(data)
        data.attrs.update(attrs)

    original_coords = data.coords

    trace("Determining dimensions.")
    if "phi" not in data.dims:
        raise ValueError("The data must have a phi coordinate.")
    trace("Replacing dummy coordinates with index-like ones.")
    removed = [d for d in data.dims if d not in ["eV", "phi"]]
    data = data.transpose(*(["eV", "phi"] + removed))
    converted_dims = data.dims

    restore_index_like_coordinates = {r: data.coords[r].values for r in removed}
    new_index_like_coordinates = {r: np.arange(len(data.coords[r].values)) for r in removed}
    data = data.assign_coords(**new_index_like_coordinates)

    converter = ConvertTrapezoidalCorrection(data, converted_dims, corners=corners)
    converted_coordinates = converter.get_coordinates()

    trace("Calling convert_coordinates")
    result = convert_coordinates(
        data,
        converted_coordinates,
        {
            "dims": data.dims,
            "transforms": dict(zip(data.dims, [converter.conversion_for(d) for d in data.dims])),
        },
        trace=trace,
    )

    trace("Reassigning index-like coordinates.")
    result = result.assign_coords(**restore_index_like_coordinates)
    result = result.assign_coords(
        **{c: v for c, v in original_coords.items() if c not in result.coords}
    )
    result = result.assign_attrs(data.attrs)
    return result
