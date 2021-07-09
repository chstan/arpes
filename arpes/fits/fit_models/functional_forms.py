"""Common implementations of peaks, backgrounds for other models."""
import numpy as np
from scipy.special import erfc  # pylint: disable=no-name-in-module

__all__ = [
    "gaussian",
    "affine_bkg",
    "lorentzian",
    "twolorentzian",
    "fermi_dirac",
    "fermi_dirac_affine",
    "gstepb",
    "gstep",
    "gstep_stdev",
    "band_edge_bkg",
    "g",
]


def affine_bkg(x: np.ndarray, lin_bkg=0, const_bkg=0) -> np.ndarray:
    """An affine/linear background.

    Args:
        x:
        lin_bkg:
        const_bkg:

    Returns:
        Background of the form
          lin_bkg * x + const_bkg
    """
    return lin_bkg * x + const_bkg


def gaussian(x, center=0, sigma=1, amplitude=1):
    """Some constants are absorbed here into the amplitude factor."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def g(x, mu=0, sigma=0.1):
    """TODO, unify this with the standard Gaussian definition because it's gross."""
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(1 / 2) * ((x - mu) / sigma) ** 2)


def lorentzian(x, gamma, center, amplitude):
    """A straightforward Lorentzian."""
    return amplitude * (1 / (2 * np.pi)) * gamma / ((x - center) ** 2 + (0.5 * gamma) ** 2)


def fermi_dirac(x, center=0, width=0.05, scale=1):
    """Fermi edge, with somewhat arbitrary normalization."""
    return scale / (np.exp((x - center) / width) + 1)


def gstepb(x, center=0, width=1, erf_amp=1, lin_bkg=0, const_bkg=0):
    """Fermi function convoled with a Gaussian together with affine background.

    This accurately represents low temperature steps where thermal broadening is
    less substantial than instrumental resolution.

    Args:
        x: value to evaluate function at
        center: center of the step
        width: width of the step
        erf_amp: height of the step
        lin_bkg: linear background slope
        const_bkg: constant background

    Returns:
        The step edge.
    """
    dx = x - center
    return const_bkg + lin_bkg * np.min(dx, 0) + gstep(x, center, width, erf_amp)


def gstep(x, center=0, width=1, erf_amp=1):
    """Fermi function convolved with a Gaussian.

    Args:
        x: value to evaluate fit at
        center: center of the step
        width: width of the step
        erf_amp: height of the step

    Returns:
        The step edge.
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(1.66511 * dx / width)


def band_edge_bkg(
    x, center=0, width=0.05, amplitude=1, gamma=0.1, lor_center=0, offset=0, lin_bkg=0, const_bkg=0
):
    """Lorentzian plus affine background multiplied into fermi edge with overall offset."""
    return (lorentzian(x, gamma, lor_center, amplitude) + lin_bkg * x + const_bkg) * fermi_dirac(
        x, center, width
    ) + offset


def fermi_dirac_affine(x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1):
    """Fermi step edge with a linear background above the Fermi level."""
    # Fermi edge with an affine background multiplied in
    return (scale + lin_bkg * x) / (np.exp((x - center) / width) + 1) + const_bkg


def gstep_stdev(x, center=0, sigma=1, erf_amp=1):
    """Fermi function convolved with a Gaussian.

    Args:
        x: value to evaluate fit at
        center: center of the step
        sigma: width of the step
        erf_amp: height of the step
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(np.sqrt(2) * dx / sigma)


def twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg):
    """A double lorentzian model.

    This is typically not necessary, as you can use the
    + operator on the Model instances. For instance `LorentzianModel() + LorentzianModel(prefix='b')`.

    This mostly exists for people that prefer to do things the "Igor Way".

    Args:
        x
        gamma
        t_gamma
        center
        t_center
        amp
        t_amp
        lin_bkg
        const_bkg

    Returns:
        A two peak structure.
    """
    L1 = lorentzian(x, gamma, center, amp)
    L2 = lorentzian(x, t_gamma, t_center, t_amp)
    AB = affine_bkg(x, lin_bkg, const_bkg)
    return L1 + L2 + AB
