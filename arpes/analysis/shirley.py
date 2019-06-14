import warnings
import numpy as np

from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = (
    'calculate_shirley_background',
    'calculate_shirley_background_full_range',
    'remove_shirley_background',
)


def remove_shirley_background(xps: DataType, **kwargs):
    xps = normalize_to_spectrum(xps)
    return xps - calculate_shirley_background(xps, **kwargs)


def calculate_shirley_background_full_range(xps: DataType, eps=1e-7, max_iters=50, n_samples=5):
    """
    Calculates a shirley background in the range of `energy_slice` according to:

    S(E) = I(E_right) + k * (A_right(E)) / (A_left(E) + A_right(E))

    Typically

    k := I(E_right) - I(E_left)

    The iterative method is continued so long as the total background is not converged to relative error
    `eps`.

    The method continues for a maximum number of iterations `max_iters`.

    In practice, what we can do is to calculate the cumulative sum of the data along the energy axis of
    both the data and the current estimate of the background
    :param xps:
    :param eps:
    :return:
    """

    xps = normalize_to_spectrum(xps)
    background = xps.copy(deep=True)
    cumulative_xps = np.cumsum(xps.values)
    total_xps = np.sum(xps.values)

    rel_error = np.inf

    i_left = np.mean(xps.values[:n_samples])
    i_right = np.mean(xps.values[-n_samples:])

    i = 0

    k = i_left - i_right
    for i in range(max_iters):
        cumulative_background = np.cumsum(background.values)
        total_background = np.sum(background.values)

        new_bkg = background.copy(deep=True)

        for i in range(len(new_bkg)):
            new_bkg.values[i] = i_right + k * (
                (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i])) / (total_xps - total_background + 1e-5)
            )

        rel_error = np.abs(np.sum(new_bkg.values) - total_background) / (total_background)

        background = new_bkg

        if rel_error < eps:
            break

    if (i + 1) == max_iters:
        warnings.warn('Shirley background calculation did not converge ' +
                      'after {} steps with relative error {}!'.format(
            max_iters, rel_error
        ))

    return background


def calculate_shirley_background(xps: DataType, energy_range: slice=None, eps=1e-7, max_iters=50, n_samples=5):
    """
    Calculates a shirley background iteratively over the full energy range `energy_range`.
    :param xps:
    :param energy_range:
    :param eps:
    :param max_iters:
    :return:
    """
    if energy_range is None:
        energy_range = slice(None, None)

    xps = normalize_to_spectrum(xps)
    xps_for_calc = xps.sel(eV=energy_range)

    bkg = calculate_shirley_background_full_range(xps_for_calc, eps, max_iters)
    full_bkg = xps * 0

    left_idx = np.searchsorted(full_bkg.eV.values, bkg.eV.values[0], side='left')
    right_idx = left_idx + len(bkg)

    full_bkg.values[:left_idx] = bkg.values[0]
    full_bkg.values[left_idx:right_idx] = bkg.values
    full_bkg.values[right_idx:] = bkg.values[-1]

    return full_bkg