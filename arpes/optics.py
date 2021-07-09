"""Some utilities for optics and optical design.

This and the utilities in arpes.laser should maybe be grouped into a separate place.

We don't do any astigmatism aware calculations here, which might be of practical utility.

Things to offer in the future:

1. Calculating beam characteristics (beam divergence, focus, etc) from a sequence of
   knife edge tests or beam profiler images in order to facilitate common tasks like beam
   collimation, focusing, or preparing a new Tr-ARPES setup.
2. Nonlinear optics utilities including damage thresholds to allow simpler design of
   harmonic generation for Tr-ARPES.
"""

# pylint: disable=invalid-name,no-value-for-parameter

import numpy as np

__all__ = (
    "waist",
    "waist_R",
    "rayleigh_range",
    "lens_transfer",
    "magnification",
    "waist_from_divergence",
    "waist_from_rr",
)


def waist(wavelength: float, z: float, z_R: float) -> float:
    """Calculates the waist size from the measurements at a distance from the waist."""
    raise NotImplementedError


def waist_R(waist_0: float, m_squared: float = 1.0) -> float:
    """Calculates the width of t he beam a distance from the waist."""
    raise NotImplementedError
    # return np.sqrt(m_squared) * waist()


def waist_from_rr(wavelength: float, rayleigh_rng: float) -> float:
    """Calculates the waist parameters from the Rayleigh range."""
    return np.sqrt((wavelength * rayleigh_rng) / np.pi)


def rayleigh_range(wavelength: float, beam_waist: float, m_squared: float = 1.0) -> float:
    """Calculates the Rayleigh range from beam parameters at the waist."""
    return np.pi * (beam_waist ** 2) / (m_squared * wavelength)


def lens_transfer(s: float, f: float, beam_rayleigh_range: float, m_squared: float = 1.0) -> float:
    """Lens transfer calculation. Calculates the location of the image.

    Args:
        s: The object distance.
        f: The focal length of the lens.
        beam_rayleigh_range: The Rayleigh range for the beam, uncorrected for the beam quality factor.
        m_squared: The beam quality factor.

    Returns:
        The distance to the image.
    """
    t = 1 / f - 1 / (s + (beam_rayleigh_range / m_squared) ** 2 / (s - f))
    return 1 / t


def waist_from_divergence(wavelength: float, half_angle_divergence: float) -> float:
    """Calculates the waist from the beam's half angle divergence and wavelength."""
    return wavelength / (np.pi * half_angle_divergence)


def magnification(s: float, f: float, beam_rayleigh_range: float, m_squared: float = 1.0) -> float:
    """Calculates the magnification offered by a lens system.

    Args:
        s: The object distance.
        f: The focal length of the lens.
        beam_rayleigh_range: The Rayleigh range for the beam, uncorrected for the beam quality factor.
        m_squared: The beam quality factor.

    Returns:
        The magnificiation of a lens.
    """
    return 1 / np.sqrt((1 - s / f) ** 2 + (beam_rayleigh_range / f / m_squared) ** 2)
