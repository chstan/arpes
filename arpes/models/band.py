"""Rudimentary band analyis code."""
import numpy as np
import scipy.ndimage.filters

import arpes.fits
import xarray as xr
from arpes.analysis.band_analysis_utils import param_getter, param_stderr_getter

__all__ = [
    "Band",
    "MultifitBand",
    "VoigtBand",
    "BackgroundBand",
]


class Band:
    """Representation of an ARPES band which supports some calculations after fitting."""

    def __init__(self, label, display_label=None, data=None):
        """Set the data but don't perform any calculation eagerly."""
        self.label = label
        self._display_label = display_label
        self._data = data

    @property
    def display_label(self):
        """The label shown on plotting tools."""
        return self._display_label or self.label

    @display_label.setter
    def display_label(self, value):
        """Set the display used for indicating the band on plotting tools."""
        self._display_label = value

    @property
    def velocity(self):
        """The band velocity."""
        spacing = float(self.coords[self.dims[0]][1] - self.coords[self.dims[0]][0])

        def embed_nan(values, padding):
            embedded = np.ndarray((values.shape[0] + 2 * padding,))
            embedded[:] = float("nan")
            embedded[padding:-padding] = values
            return embedded

        raw_values = embed_nan(np.copy(self.center.values), 50)

        masked = np.copy(raw_values)
        masked[raw_values != raw_values] = 0

        nan_mask = np.copy(raw_values) * 0 + 1
        nan_mask[raw_values != raw_values] = 0

        sigma = 0.1 / spacing
        nan_mask = scipy.ndimage.gaussian_filter(nan_mask, sigma, mode="mirror")
        masked = scipy.ndimage.gaussian_filter(masked, sigma, mode="mirror")

        return xr.DataArray(np.gradient(masked / nan_mask, spacing)[50:-50], self.coords, self.dims)

    @property
    def fermi_velocity(self):
        """The band velocity evaluated at the Fermi level."""
        return self.velocity.sel(eV=0, method="nearest")

    @property
    def band_width(self):
        """The width along the band."""
        return None

    def band_energy(self, coordinates):
        """The energy coordinate along the band."""
        pass

    @property
    def self_energy(self):
        """Calculates the self energy along the band."""
        return None

    @property
    def fit_cls(self):
        """Describes which fit class to use for band fitting, default Lorentzian."""
        return arpes.fits.LorentzianModel

    def get_dataarray(self, var_name, clean=True):
        """Converts the underlying data into an array representation."""
        if not clean:
            return self._data[var_name].values

        output = np.copy(self._data[var_name].values)
        output[self._data[var_name + "_stderr"].values > 0.01] = float("nan")

        return xr.DataArray(
            output,
            self._data[var_name].coords,
            self._data[var_name].dims,
        )

    @property
    def center(self):
        """Gets the peak location along the band."""
        return self.get_dataarray("center")

    @property
    def center_stderr(self):
        """Gets the peak location stderr along the band."""
        return self.get_dataarray("center_stderr", False)

    @property
    def sigma(self):
        """Gets the peak width along the band."""
        return self.get_dataarray("sigma", True)

    @property
    def amplitude(self):
        """Gets the peak amplitude along the band."""
        return self.get_dataarray("amplitude", True)

    @property
    def indexes(self):
        """Fetches the indices of the originating data (after fit reduction)."""
        return self._data.center.indexes

    @property
    def coords(self):
        """Fetches the coordinates of the originating data (after fit reduction)."""
        return self._data.center.coords

    @property
    def dims(self):
        """Fetches the dimensions of the originating data (after fit reduction)."""
        return self._data.center.dims


class MultifitBand(Band):
    """Convenience class that reimplements reading data out of a composite fit result."""

    def get_dataarray(self, var_name, clean=True):
        """Converts the underlying data into an array representation."""
        full_var_name = self.label + var_name

        if "stderr" in full_var_name:
            return self._data.G.map(param_stderr_getter(full_var_name.split("_stderr")[0]))

        return self._data.G.map(param_getter(full_var_name))


class VoigtBand(Band):
    """Uses a Voigt lineshape."""

    @property
    def fit_cls(self):
        """Fit using `arpes.fits.VoigtModel`."""
        return arpes.fits.VoigtModel


class BackgroundBand(Band):
    """Uses a Gaussian lineshape."""

    @property
    def fit_cls(self):
        """Fit using `arpes.fits.GaussianModel`."""
        return arpes.fits.GaussianModel
