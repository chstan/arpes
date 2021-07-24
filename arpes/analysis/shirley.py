"""Contains routines for calculating and removing the classic Shirley background."""
import warnings

import numpy as np
import xarray as xr

from arpes.provenance import update_provenance
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = (
    "calculate_shirley_background",
    "calculate_shirley_background_full_range",
    "remove_shirley_background",
)


@update_provenance("Remove Shirley background")
def remove_shirley_background(xps: DataType, **kwargs) -> DataType:
    """Calculates and removes a Shirley background from a spectrum.

    Only the background corrected spectrum is retrieved.

    Args:
        xps: The input array.
        kwargs: Parameters to feed to the background estimation routine.

    Returns:
        The the input array with a Shirley background subtracted.
    """
    xps = normalize_to_spectrum(xps)
    return xps - calculate_shirley_background(xps, **kwargs)


def _calculate_shirley_background_full_range(
    xps: np.ndarray, eps=1e-7, max_iters=50, n_samples=5
) -> np.ndarray:
    """Core routine for calculating a Shirley background on np.ndarray data."""
    background = np.copy(xps)
    cumulative_xps = np.cumsum(xps, axis=0)
    total_xps = np.sum(xps, axis=0)

    rel_error = np.inf

    i_left = np.mean(xps[:n_samples], axis=0)
    i_right = np.mean(xps[-n_samples:], axis=0)

    iter_count = 0

    k = i_left - i_right
    for iter_count in range(max_iters):
        cumulative_background = np.cumsum(background, axis=0)
        total_background = np.sum(background, axis=0)

        new_bkg = np.copy(background)

        for i in range(len(new_bkg)):
            new_bkg[i] = i_right + k * (
                (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                / (total_xps - total_background + 1e-5)
            )

        rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

        background = new_bkg

        if np.any(rel_error < eps):
            break

    if (iter_count + 1) == max_iters:
        warnings.warn(
            "Shirley background calculation did not converge "
            + "after {} steps with relative error {}!".format(max_iters, rel_error)
        )

    return background


@update_provenance("Calculate full range Shirley background")
def calculate_shirley_background_full_range(
    xps: DataType, eps=1e-7, max_iters=50, n_samples=5
) -> DataType:
    """Calculates a shirley background.

    The background is defined according to:

    S(E) = I(E_right) + k * (A_right(E)) / (A_left(E) + A_right(E))

    Typically

    k := I(E_right) - I(E_left)

    The iterative method is continued so long as the total background is not converged to relative error
    `eps`.

    The method continues for a maximum number of iterations `max_iters`.

    In practice, what we can do is to calculate the cumulative sum of the data along the energy axis of
    both the data and the current estimate of the background

    Args:
        xps: The input data.
        eps: Convergence parameter.
        max_iters: The maximum number of iterations to allow before convengence.
        n_samples: The number of samples to use at the boundaries of the input data.

    Returns:
        A monotonic Shirley backgruond over the entire energy range.
    """
    xps = normalize_to_spectrum(xps).copy(deep=True)
    core_dims = [d for d in xps.dims if d != "eV"]

    return xr.apply_ufunc(
        _calculate_shirley_background_full_range,
        xps,
        eps,
        max_iters,
        n_samples,
        input_core_dims=[core_dims, [], [], []],
        output_core_dims=[core_dims],
        exclude_dims=set(core_dims),
        vectorize=False,
    )


@update_provenance("Calculate limited range Shirley background")
def calculate_shirley_background(
    xps: DataType, energy_range: slice = None, eps=1e-7, max_iters=50, n_samples=5
) -> DataType:
    """Calculates a shirley background iteratively over the full energy range `energy_range`.

    Uses `calculate_shirley_background_full_range` internally.

    Outside the indicated range, the background is extrapolated as a constant from
    the nearest in-range value.

    Args:
        xps: The input data.
        energy_range: A slice with the energy range to be used.
        eps: Convergence parameter.
        max_iters: The maximum number of iterations to allow before convengence.
        n_samples: The number of samples to use at the boundaries of the input data.

    Returns:
        A monotonic Shirley backgruond over the entire energy range.
    """
    if energy_range is None:
        energy_range = slice(None, None)

    xps = normalize_to_spectrum(xps)
    xps_for_calc = xps.sel(eV=energy_range)

    bkg = calculate_shirley_background_full_range(xps_for_calc, eps, max_iters, n_samples)
    bkg = bkg.transpose(*xps.dims)
    full_bkg = xps * 0

    left_idx = np.searchsorted(full_bkg.eV.values, bkg.eV.values[0], side="left")
    right_idx = left_idx + len(bkg)

    full_bkg.values[:left_idx] = bkg.values[0]
    full_bkg.values[left_idx:right_idx] = bkg.values
    full_bkg.values[right_idx:] = bkg.values[-1]

    return full_bkg
