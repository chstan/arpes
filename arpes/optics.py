"""
Some utilities for optics and optical design. This and the utilities in arpes.laser
should maybe be grouped into a separate place.

We don't do any astigmatism aware calculations here, which might be of practical utility.

Things to offer in the future:

1. Calculating beam characteristics (beam divergence, focus, etc) from a sequence of
   knife edge tests or beam profiler images in order to facilitate common tasks like beam
   collimation, focusing, or preparing a new Tr-ARPES setup.
2. Nonlinear optics utilities including damage thresholds to allow simpler design of
   harmonic generation for Tr-ARPES.
"""

import numpy as np

__all__ = ('waist', 'waist_R', 'rayleigh_range', 'lens_transfer', 'magnification', 'waist_from_divergence', 'waist_from_rr')


def waist(wavelength, z, z_R):

    raise NotImplementedError('waist not implemented')


def waist_R(waist_0, m_squared=1):
    return np.sqrt(m_squared) * waist()


def waist_from_rr(wavelength, rayleigh_range):
    return np.sqrt((wavelength * rayleigh_range) / np.pi)


def rayleigh_range(wavelength, waist, m_squared=1):
    return np.pi * (waist ** 2) / (m_squared * wavelength)


def lens_transfer(s, f, rayleigh_range, m_squared=1):
    """
    Produces s''
    :param s:
    :param f:
    :param f_p:
    :param m_squared:
    :return:
    """

    t = 1/f - 1/(s + (rayleigh_range/m_squared) ** 2/(s - f))
    return 1 / t


def waist_from_divergence(wavelength, half_angle_divergence):
    return wavelength / (np.pi * half_angle_divergence)


def magnification(s, f, rayleigh_range, m_squared=1):
    """
    Calculates the magnification offered by a lens system.
    :param s:
    :param f:
    :param rayleigh_range:
    :param m_squared:
    :return:
    """
    return 1/np.sqrt((1 - s/f) ** 2 + (rayleigh_range/f/m_squared)**2)