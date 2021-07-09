"""Utilities for estimating quantities of interest when using a laser for photoemission."""

import pint
from arpes.config import ureg
from typing import Optional

__all__ = ("electrons_per_pulse", "electrons_per_pulse_mira")

mira_frequency = 54.3 / ureg.microsecond


def electrons_per_pulse(
    photocurrent: pint.Quantity,
    repetition_rate: Optional[pint.Quantity],
    division_ratio: int = 1,
) -> float:
    """Calculates the number of photoemitted electrons per pulse for pulsed lasers.

    Either the pulse_rate or the division_ratio and the base_repetition_rate should be
    specified.

    Args:
        photocurrent: The photocurrent in `pint` current units (i.e. amps).
        repetition_rate: The repetition rate for the laser, in `pint` frequency units.
        division_ratio: The division_ratio for a pulse-picked laser. Optionally modifies the
          repetition rate used for the calculation.

    Returns:
        The expectation of the number of electrons emitted per pulse of the laser.
    """
    repetition_rate /= division_ratio

    eles_per_attocoulomb = 6.2415091
    atto_coulombs = (photocurrent / repetition_rate).to("attocoulomb")

    return (atto_coulombs * eles_per_attocoulomb).magnitude


def electrons_per_pulse_mira(photocurrent: pint.Quantity, division_ratio: int = 1) -> float:
    """Specific case of `electrons_per_pulse` for Mira oscillators."""
    return electrons_per_pulse(photocurrent, mira_frequency, division_ratio)
