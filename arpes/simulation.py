import numpy as np
import scipy.signal as sig
import scipy
import xarray as xr

from arpes.constants import K_BOLTZMANN_MEV_KELVIN

__all__ = (
    # sampling utilities
    'point_cloud_to_arr',
    'apply_psf_to_point_cloud',
    'sample_from_distribution',

    # Spectral Function representation
    'SpectralFunction',

    # Composable detector effects, to simulate the real response of
    # an ARPES detector
    'DetectorEffect',

    # implementations of particular spectral functions
    'SpectralFunctionBSSCO', 'SpectralFunctionMFL',
    'SpectralFunctionPhaseCoherent',

    # Particular Detector Models
    'NonlinearDetectorEffect',
    'FixedModeDetectorEffect',
    'DustDetectorEffect',
    'TrapezoidalDetectorEffect',
    'WindowedDetectorEffect',
)


class DetectorEffect(object):
    def __call__(self, spectrum):
        """
        By default, the identity function
        :param spectrum:
        :return:
        """
        return spectrum


class NonlinearDetectorEffect(DetectorEffect):
    def __init__(self, gamma=None, nonlinearity=None):
        self.gamma = gamma
        self.nonlinearity = nonlinearity

    def __call__(self, spectrum):
        if self.gamma is not None:
            return spectrum ** self.gamma

        raise NotImplementedError(
            'Nonlinearity lookup tables are not yet supported.')


class FixedModeDetectorEffect(DetectorEffect):
    def __init__(self, spacing=None, periodic='hex', detector_efficiency=None):
        if spacing is None:
            spacing = 5 # Five pixel average spacing
        self.spacing = spacing
        self.periodic = periodic
        self._cached_pore_structure = None

    @property
    def detector_imprint(self):
        raise NotImplementedError()

    def __call__(self, spectrum):
        # will fail if we do not have the right size
        return self.detector_imprint * spectrum


class DustDetectorEffect(DetectorEffect):
    pass


class TrapezoidalDetectorEffect(DetectorEffect):
    pass


class WindowedDetectorEffect(DetectorEffect):
    pass


def cloud_to_arr(point_cloud, shape):
    """
    Converts a point cloud (list of xy pairs) to an array representation.
    Uses linear interpolation for points that have non-integral coordinates.
    :param point_cloud:
    :param shape:
    :return:
    """
    cloud_as_image = np.zeros(shape)

    for x, y in zip(*point_cloud):
        frac_low_x = 1 - (x - np.floor(x))
        frac_low_y = 1 - (y - np.floor(y))
        sx, sy = shape
        cloud_as_image[int(np.floor(x)) % sx][int(np.floor(y)) % sy] += frac_low_x * frac_low_y
        cloud_as_image[(int(np.floor(x)) + 1) % sx][int(np.floor(y)) % sy] += (1 - frac_low_x) * frac_low_y
        cloud_as_image[int(np.floor(x)) % sx][(int(np.floor(y)) + 1) % sy] += frac_low_x * (1 - frac_low_y)
        cloud_as_image[(int(np.floor(x)) + 1) % sx][(int(np.floor(y)) + 1) % sy] += (1 - frac_low_x) * (1 - frac_low_y)

    return cloud_as_image


def apply_psf_to_point_cloud(point_cloud, shape, sigma=None):
    if sigma is None:
        sigma = (10, 3)
    as_img = cloud_to_arr(point_cloud, shape)

    return scipy.ndimage.gaussian_filter(as_img, sigma=sigma, order=0, mode='reflect')


def sample_from_distribution(distribution, N=5000):
    cdf_rows = np.cumsum(np.sum(distribution.values, axis=1))
    norm_rows = np.cumsum(distribution.values / np.expand_dims(np.sum(distribution.values, axis=1), axis=1), axis=1)

    total = np.sum(distribution.values)

    sample_xs = np.searchsorted(cdf_rows, np.random.random(N,) * total)
    sample_ys_rows = norm_rows[sample_xs, :]

    sample_ys = []
    rys = np.random.random(N, )
    for ry, row_y in zip(rys, sample_ys_rows):
        sample_ys.append(np.searchsorted(row_y, ry))

    return (1. * sample_xs + np.random.random(N, )), (1. * np.array(sample_ys) + np.random.random(N, ))


class SpectralFunction(object):
    def fermi_dirac(self, omega):
        return 1 / (np.exp(omega / (K_BOLTZMANN_MEV_KELVIN * self.T)) + 1)

    def __init__(self, k=None, omega=None, T=None):
        if T is None:
            T = 20
        if k is None:
            k = np.linspace(-200, 200, 800)
        elif len(k) == 3:
            k = np.linspace(*k)

        if omega is None:
            omega = np.linspace(-1000, 1000, 2000)
        elif len(omega) == 3:
            omega = np.linspace(*omega)

        self.T = T
        self.omega = omega
        self.k = k

    def imag_self_energy(self):
        raise NotImplementedError()

    def real_self_energy(self):
        """
        Default to Kramers-Kronig
        """
        return np.imag(sig.hilbert(self.imag_self_energy()))

    def self_energy(self):
        return self.real_self_energy() + 1.j * self.imag_self_energy()

    def bare_band(self):
        return 3 * self.k

    def sampled_spectral_function(self, n_electrons=50000, n_cycles=1, psf=None):
        if psf is None:
            psf = (7, 3)

        spectral = self.measured_spectral_function()
        sampled = [
            apply_psf_to_point_cloud(sample_from_distribution(spectral, N=n_electrons), spectral.values.shape, sigma=psf)
            for _ in range(n_cycles)
        ]

        new_coords = dict(spectral.coords)
        new_coords['cycle'] = np.array(range(n_cycles))
        return xr.DataArray(np.stack(sampled, axis=-1), coords=new_coords, dims=list(spectral.dims) + ['cycle'])

    def measured_spectral_function(self):
        spectral = self.occupied_spectral_function()
        return spectral

    def occupied_spectral_function(self):
        spectral = self.spectral_function()
        spectral.values = spectral.values * np.expand_dims(self.fermi_dirac(self.omega), axis=1)
        return spectral

    def spectral_function(self):
        self_energy = self.self_energy()
        imag_self_energy = np.imag(self_energy)
        real_self_energy = np.real(self_energy)

        bare = self.bare_band()
        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        numerator = np.outer(np.abs(imag_self_energy), np.ones(shape=bare.shape))
        data = numerator / ((full_omegas - np.expand_dims(bare, axis=0) -
                             np.expand_dims(real_self_energy, axis=1)) ** 2 +
                            np.expand_dims(imag_self_energy ** 2, axis=1))
        return xr.DataArray(data, coords={'k': self.k, 'omega': self.omega}, dims=['omega', 'k'])


class SpectralFunctionMFL(SpectralFunction):
    """
    Implements the Marginal Fermi Liquid spectral function, more or less.
    """

    def __init__(self, k=None, omega=None, T=None, a=None, b=None):
        if a is None:
            a = 10

        if b is None:
            b = 1

        super().__init__(k, omega, T)

        self.a = a
        self.b = b

    def imag_self_energy(self):
        return np.sqrt((self.a + self.b * self.omega) ** 2 + self.T ** 2)


class SpectralFunctionBSSCO(SpectralFunction):
    """
    Implements the spectral function for BSSCO as reported in PhysRevB.57.R11093 and explored in
    https://arxiv.org/pdf/1707.02305.pdf
    """

    def __init__(self, k=None, omega=None, T=None, delta=None, gamma_s=None, gamma_p=None):
        if delta is None:
            delta = 50
        if gamma_s is None:
            gamma_s = 30
        if gamma_p is None:
            gamma_p = 10

        self.delta = delta
        self.gamma_s = gamma_s
        self.gamma_p = gamma_p
        super().__init__(k, omega, T)

    def self_energy(self):
        shape = (len(self.omega), len(self.k))

        g_one = -1.j * self.gamma_s * np.ones(shape=shape)
        bare = self.bare_band()

        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        return g_one + (self.delta ** 2) / (full_omegas + bare + 1.j * self.gamma_p)

    def spectral_function(self):
        self_energy = self.self_energy()
        imag_self_energy = np.imag(self_energy)
        real_self_energy = np.real(self_energy)

        bare = self.bare_band()
        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        numerator = np.abs(imag_self_energy)
        data = numerator / ((full_omegas - np.expand_dims(bare, axis=0) -
                             real_self_energy) ** 2 +
                            imag_self_energy ** 2)
        return xr.DataArray(data, coords={'k': self.k, 'omega': self.omega}, dims=['omega', 'k'])


class SpectralFunctionPhaseCoherent(SpectralFunctionBSSCO):
    def self_energy(self):
        shape = (len(self.omega), len(self.k))

        g_one = -1.j * self.gamma_s * np.ones(shape=shape) * np.sqrt(
            (1 + 0.0005 * np.expand_dims(self.omega, axis=1) ** 2))
        bare = self.bare_band()

        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        se = (g_one + (self.delta ** 2) / (full_omegas + bare + 1.j * self.gamma_p))
        ise = np.imag(se)
        rse = np.imag(sig.hilbert(ise, axis=0))

        return se + 3 * (np.random.random(se.shape) + np.random.random(se.shape) * 1.j)
