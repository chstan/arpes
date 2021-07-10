"""Phenomenological models and detector simulation for accurate modelling of ARPES data.

Currently we offer relatively rudimentary detecotor modeling, mostly providing
only nonlinearity and some stubs, but a future release will provide reasonably
accurate modeling of the trapezoidal effect in hemispherical analyzers,
fixed mode artifacts, dust, and more.

Additionally we offer the ability to model the detector response at the
level of individual electron events using a point spread or more complicated
response. This allows the creation of reasonably realistic spectra for testing
new analysis techniques or working on machine learning based approaches that must
be robust to the shortcomings of actual ARPES data.
"""

from dataclasses import dataclass
import numpy as np
import scipy
import scipy.signal as sig

import xarray as xr
from arpes.constants import K_BOLTZMANN_MEV_KELVIN

from typing import Dict, Any, Optional, Tuple

__all__ = (
    # sampling utilities
    "cloud_to_arr",
    "apply_psf_to_point_cloud",
    "sample_from_distribution",
    # Spectral Function representation
    "SpectralFunction",
    # Composable detector effects, to simulate the real response of
    # an ARPES detector
    "DetectorEffect",
    # implementations of particular spectral functions
    "SpectralFunctionBSSCO",
    "SpectralFunctionMFL",
    "SpectralFunctionPhaseCoherent",
    # Particular Detector Models
    "NonlinearDetectorEffect",
    "FixedModeDetectorEffect",
    "DustDetectorEffect",
    "TrapezoidalDetectorEffect",
    "WindowedDetectorEffect",
)


class DetectorEffect:
    """Detector effects are callables that map a spectrum into a new transformed one.

    This might be used to imprint the image of a grid,
    dust, or impose detector nonlinearities.
    """

    def __call__(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Applies the detector effect to a spectrum.

        By default, apply the identity function.

        Args:
            spectrum: The input spectrum before modification.

        Returns:
            The data with this effect applied.
        """
        return spectrum


@dataclass
class NonlinearDetectorEffect(DetectorEffect):
    """Implements power law detector nonlinearities."""

    gamma: Optional[float] = 1.0

    def __call__(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Applies the detector effect to a spectrum.

        The effect is modeled by letting the output intensity (I_out) be equal to
        the input intensity (I_in) to a fixed power. I.e. I_out[i, j] = I_in[i, j]^gamma.

        Args:
            spectrum: The input spectrum before modification.

        Returns:
            The data with this effect applied.
        """
        if self.gamma is not None:
            return spectrum ** self.gamma

        raise NotImplementedError("Nonlinearity lookup tables are not yet supported.")


@dataclass
class FixedModeDetectorEffect(DetectorEffect):
    """Implements a grid or pore structure of an MCP or field termination mesh.

    Talk to Danny or Sam about getting hyperuniform point cloud distributions to use
    for the pore structure. Otherwise, we can just use a sine-grid.

    Attributes:
        spacing: The pixel periodicity of the pores.
        periodic: The grid type to use for the pores. One of ["hex"].
    """

    spacing: float = 5.0
    periodic: str = "hex"

    _cached_pore_structure = None

    @property
    def detector_imprint(self) -> xr.DataArray:
        """Provides the transmission factor for the grid on the spectrometer or "imprint"."""
        raise NotImplementedError

    def __call__(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Applies the detector effect to a spectrum.

        Args:
            spectrum: The input spectrum before modification.

        Returns:
            The data with this effect applied.
        """
        # will fail if we do not have the right size
        return self.detector_imprint * spectrum


class DustDetectorEffect(DetectorEffect):
    """Applies aberrations in the spectrum coming from dust.

    TODO, dust.
    """


class TrapezoidalDetectorEffect(DetectorEffect):
    """Applies trapezoidal detector windowing.

    TODO model that phi(pixel) is also a function of binding energy,
    i.e. that the detector has severe aberrations at low photoelectron
    kinetic energy (high retardation ratio).
    """


class WindowedDetectorEffect(DetectorEffect):
    """TODO model the finite width of the detector window as recorded on a camera."""


def cloud_to_arr(point_cloud, shape) -> np.ndarray:
    """Converts a point cloud (list of xy pairs) to an array representation.

    Uses linear interpolation for points that have non-integral coordinates.

    Args:
        point_cloud: The sampled set of electrons.
        shape: The shape of the desired output array.

    Returns:
        An array with the electron arrivals smeared into it.
    """
    cloud_as_image = np.zeros(shape)

    for x, y in zip(*point_cloud):
        frac_low_x = 1 - (x - np.floor(x))
        frac_low_y = 1 - (y - np.floor(y))
        shape_x, shape_y = shape
        cloud_as_image[int(np.floor(x)) % shape_x][int(np.floor(y)) % shape_y] += (
            frac_low_x * frac_low_y
        )
        cloud_as_image[(int(np.floor(x)) + 1) % shape_x][int(np.floor(y)) % shape_y] += (
            1 - frac_low_x
        ) * frac_low_y
        cloud_as_image[int(np.floor(x)) % shape_x][
            (int(np.floor(y)) + 1) % shape_y
        ] += frac_low_x * (1 - frac_low_y)
        cloud_as_image[(int(np.floor(x)) + 1) % shape_x][(int(np.floor(y)) + 1) % shape_y] += (
            1 - frac_low_x
        ) * (1 - frac_low_y)

    return cloud_as_image


def apply_psf_to_point_cloud(point_cloud, shape, sigma: Tuple[int, int] = (10, 3)) -> np.ndarray:
    """Takes a point cloud and turns it into a broadened spectrum.

    Samples are drawn individually and smeared by a
    gaussian PSF given through the `sigma` parameter. Their net contribution
    as an integrated image is returned.

    In the future, we should also allow for specifying a particular PSF.

    Args:
        point_cloud: The sampled set of electrons.
        shape: The shape of the desired output array.
        sigma: The broadening to apply, in pixel units.

    Returns:
        An array with the electron arrivals smeared into it.
    """
    as_img = cloud_to_arr(point_cloud, shape)

    return scipy.ndimage.gaussian_filter(as_img, sigma=sigma, order=0, mode="reflect")


def sample_from_distribution(distribution: np.ndarray, N: int = 5000) -> np.ndarray:
    """Samples events from a probability distribution.

    Given a probability distribution in ND modeled by an array providing the PDF,
    sample individual events coming from this PDF.

    Args:
        distribution: The probability density. The probability of drawing a sample at (i, j)
          will be proportional to `distribution[i, j]`.
        N: The desired number of electrons/samples to pull from the distribution.

    Returns:
        An array with the arrival locations.
    """
    cdf_rows = np.cumsum(np.sum(distribution.values, axis=1))
    norm_rows = np.cumsum(
        distribution.values / np.expand_dims(np.sum(distribution.values, axis=1), axis=1),
        axis=1,
    )

    total = np.sum(distribution.values)

    sample_xs = np.searchsorted(
        cdf_rows,
        np.random.random(
            N,
        )
        * total,
    )
    sample_ys_rows = norm_rows[sample_xs, :]

    # take N samples between 0 and 1, which is now the normalized full range of the data
    # and find the index, this effectively samples the index in the array if it were a PDF
    sample_ys = []
    random_ys = np.random.random(
        N,
    )
    for random_y, row_y in zip(random_ys, sample_ys_rows):
        sample_ys.append(np.searchsorted(row_y, random_y))

    return (1.0 * sample_xs + np.random.random(N,)), (
        1.0 * np.array(sample_ys)
        + np.random.random(
            N,
        )
    )


class SpectralFunction:
    """Generic spectral function model for a band with self energy in the single-particle picture."""

    def digest_to_json(self) -> Dict[str, Any]:
        """Summarizes the parameters for the model to JSON."""
        return {"omega": self.omega, "temperature": self.temperature, "k": self.k}

    def fermi_dirac(self, omega: np.ndarray) -> np.ndarray:
        """Calculates the Fermi-Dirac occupation factor at energy values `omega`."""
        return 1 / (np.exp(omega / (K_BOLTZMANN_MEV_KELVIN * self.temperature)) + 1)

    def __init__(self, k=None, omega=None, temperature=20):
        """Initialize from paramters.

        Args:
            k: The momentum range for the simulation.
            omega: The energy range for the simulation.
            temperature: The temperature for the simulation.
        """
        if k is None:
            k = np.linspace(-200, 200, 800)
        elif len(k) == 3:
            k = np.linspace(*k)

        if omega is None:
            omega = np.linspace(-1000, 1000, 2000)
        elif len(omega) == 3:
            omega = np.linspace(*omega)

        self.temperature = temperature
        self.omega = omega
        self.k = k

    def imag_self_energy(self) -> np.ndarray:
        """Provides the imaginary part of the self energy."""
        return np.zeros(
            shape=self.omega.shape,
        )

    def real_self_energy(self) -> np.ndarray:
        """Defaults to using Kramers-Kronig from the imaginary self energy."""
        return np.imag(sig.hilbert(self.imag_self_energy()))

    def self_energy(self) -> np.ndarray:
        """Combines the self energy terms into a complex valued array."""
        return self.real_self_energy() + 1.0j * self.imag_self_energy()

    def bare_band(self) -> np.ndarray:
        """Provides the bare band dispersion."""
        return 3 * self.k

    def sampled_spectral_function(
        self,
        n_electrons: int = 50000,
        n_cycles: int = 1,
        psf: Optional[Tuple[int, int]] = (7, 3),
    ) -> xr.DataArray:
        """Samples electrons from the measured spectral function to calculate a detector image.

        The measured spectral function is used as a 2D density for the electrons. Samples are drawn
        and then broadened by a point spread (`psf`) modeling finite resolution detector response.

        Args:
            n_electrons: The number of electrons to draw.
            n_cycles: The number of frames to draw. `n_electrons` are drawn per cycle.
            psf: The point spread width in pixels.

        Returns:
            xr.DataArray: [description]
        """
        spectral = self.measured_spectral_function()
        sampled = [
            apply_psf_to_point_cloud(
                sample_from_distribution(spectral, N=n_electrons),
                spectral.values.shape,
                sigma=psf,
            )
            for _ in range(n_cycles)
        ]

        new_coords = dict(spectral.coords)
        new_coords["cycle"] = np.array(range(n_cycles))
        return xr.DataArray(
            np.stack(sampled, axis=-1),
            coords=new_coords,
            dims=list(spectral.dims) + ["cycle"],
        )

    def measured_spectral_function(self) -> xr.DataArray:
        """Calculates the measured spectral function under practical conditions."""
        spectral = self.occupied_spectral_function()
        return spectral

    def occupied_spectral_function(self) -> xr.DataArray:
        """Calculates the spectral function weighted by the thermal occupation."""
        spectral = self.spectral_function()
        spectral.values = spectral.values * np.expand_dims(self.fermi_dirac(self.omega), axis=1)
        return spectral

    def spectral_function(self) -> xr.DataArray:
        """Calculates the spectral function according to the self energy modification of the bare band.

        This essentially implements the classic formula for the single particle spectral function as the Lorentzian
        broadened and offset bare band.

        Returns:
            An `xr.DataArray` with the spectral function intensity in a given momentum-energy window.
        """
        self_energy = self.self_energy()
        imag_self_energy = np.imag(self_energy)
        real_self_energy = np.real(self_energy)

        bare = self.bare_band()
        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        numerator = np.outer(np.abs(imag_self_energy), np.ones(shape=bare.shape))
        data = numerator / (
            (full_omegas - np.expand_dims(bare, axis=0) - np.expand_dims(real_self_energy, axis=1))
            ** 2
            + np.expand_dims(imag_self_energy ** 2, axis=1)
        )
        return xr.DataArray(data, coords={"k": self.k, "omega": self.omega}, dims=["omega", "k"])


class SpectralFunctionMFL(SpectralFunction):  # pylint: disable=invalid-name
    """Implements the Marginal Fermi Liquid spectral function, more or less."""

    def digest_to_json(self) -> Dict[str, Any]:
        """Summarizes the parameters for the model to JSON."""
        return {
            **super().digest_to_json(),
            "a": self.a,
            "b": self.a,
        }

    def __init__(self, k=None, omega=None, temperature=None, a=10.0, b=1.0):
        """Initializes from parameters.

        Args:
            k: The momentum axis.
            omega: The energy axis.
            temperature: The temperature to use for the calculation. Defaults to None.
            a: The MFL `a` parameter. Defaults to 10.0.
            b: The MFL `b` parameter. Defaults to 1.0.
        """
        super().__init__(k, omega, temperature)

        self.a = a
        self.b = b

    def imag_self_energy(self) -> np.ndarray:
        """Calculates the imaginary part of the self energy."""
        return np.sqrt((self.a + self.b * self.omega) ** 2 + self.temperature ** 2)


class SpectralFunctionBSSCO(SpectralFunction):
    """Implements the spectral function for BSSCO as reported in PhysRevB.57.R11093.

    This spectral function is explored in the paper
    `"Collapse of superconductivity in cuprates via ultrafast quenching of phase coherence" <https://arxiv.org/pdf/1707.02305.pdf>`_.
    """

    def __init__(
        self,
        k=None,
        omega=None,
        temperature=None,
        delta=50,
        gamma_s=30,
        gamma_p=10,
    ):
        """Initializes from parameters.

        Args:
            k: The momentum axis.
            omega: The energy axis.
            temperature: The temperature to use for the calculation. Defaults to None.
            delta: The gap size.
            gamma_s: The s-wave gamma parameter.
            gamma_p: The p-wave gamma parameter.
        """
        self.delta = delta
        self.gamma_s = gamma_s
        self.gamma_p = gamma_p
        super().__init__(k, omega, temperature)

    def digest_to_json(self) -> Dict[str, Any]:
        """Summarizes the parameters for the model to JSON."""
        return {
            **super().digest_to_json(),
            "delta": self.delta,
            "gamma_s": self.gamma_s,
            "gamma_p": self.gamma_p,
        }

    def self_energy(self):
        """Calculates the self energy."""
        shape = (len(self.omega), len(self.k))

        g_one = -1.0j * self.gamma_s * np.ones(shape=shape)
        bare = self.bare_band()

        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        return g_one + (self.delta ** 2) / (full_omegas + bare + 1.0j * self.gamma_p)

    def spectral_function(self) -> xr.DataArray:
        """Calculates the spectral function according to the self energy modification of the bare band.

        This essentially implements the classic formula for the single particle spectral function as the Lorentzian
        broadened and offset bare band.

        Returns:
            An `xr.DataArray` with the spectral function intensity in a given momentum-energy window.
        """
        self_energy = self.self_energy()
        imag_self_energy = np.imag(self_energy)
        real_self_energy = np.real(self_energy)

        bare = self.bare_band()
        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        numerator = np.abs(imag_self_energy)
        data = numerator / (
            (full_omegas - np.expand_dims(bare, axis=0) - real_self_energy) ** 2
            + imag_self_energy ** 2
        )
        return xr.DataArray(data, coords={"k": self.k, "omega": self.omega}, dims=["omega", "k"])


class SpectralFunctionPhaseCoherent(SpectralFunctionBSSCO):
    """Implements the "phase coherence" model for the BSSCO spectral function."""

    def self_energy(self) -> xr.DataArray:
        """Calculates the self energy using the phase coherent BSSCO model."""
        shape = (len(self.omega), len(self.k))

        g_one = (
            -1.0j
            * self.gamma_s
            * np.ones(shape=shape)
            * np.sqrt((1 + 0.0005 * np.expand_dims(self.omega, axis=1) ** 2))
        )
        bare = self.bare_band()

        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        self_e = g_one + (self.delta ** 2) / (full_omegas + bare + 1.0j * self.gamma_p)
        imag_self_e = np.imag(self_e)
        real_self_e = np.imag(sig.hilbert(imag_self_e, axis=0))

        return self_e + 3 * (np.random.random(self_e.shape) + np.random.random(self_e.shape) * 1.0j)
