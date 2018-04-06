import numpy as np
import scipy.ndimage.filters
import xarray as xr

import arpes.fits
from arpes.analysis.band_analysis_utils import param_getter, param_stderr_getter

__all__ = ['Band', 'MultifitBand', 'VoigtBand', 'BackgroundBand', 'FermiEdgeBand',
           'AffineBackgroundBand']


class Band(object):
    def __init__(self, label, display_label=None, data=None):
        self.label = label
        self._display_label = display_label
        self._data = data

    @property
    def display_label(self):
        return self._display_label or self.label

    @display_label.setter
    def display_label(self, value):
        self._display_label = value

    @property
    def velocity(self):
        spacing = float(self.coords[self.dims[0]][1] - self.coords[self.dims[0]][0])

        def embed_nan(values, padding):
            embedded = np.ndarray((values.shape[0] + 2 * padding,))
            embedded[:] = float('nan')
            embedded[padding:-padding] = values
            return embedded

        raw_values = embed_nan(np.copy(self.center.values), 50)


        masked = np.copy(raw_values)
        masked[raw_values != raw_values] = 0

        nan_mask = np.copy(raw_values) * 0 + 1
        nan_mask[raw_values != raw_values] = 0

        sigma = 0.1 / spacing
        nan_mask = scipy.ndimage.gaussian_filter(nan_mask, sigma, mode='mirror')
        masked = scipy.ndimage.gaussian_filter(masked, sigma, mode='mirror')

        return xr.DataArray(
            np.gradient(masked / nan_mask, spacing)[50:-50],
            self.coords,
            self.dims
        )

    @property
    def fermi_velocity(self):
        return self.velocity.sel(eV=0, method='nearest')

    @property
    def band_width(self):
        return None

    def band_energy(self, coordinates):
        pass

    @property
    def self_energy(self):
        return None

    @property
    def fit_cls(self):
        return arpes.fits.LorentzianModel

    def get_dataarray(self, var_name, clean=True):
        if not clean:
            return self._data[var_name].values

        output = np.copy(self._data[var_name].values)
        output[self._data[var_name + '_stderr'].values > 0.01] = float('nan')

        return xr.DataArray(
            output,
            self._data[var_name].coords,
            self._data[var_name].dims,
        )

    @property
    def center(self, clean=True):
        return self.get_dataarray('center', clean)

    @property
    def center_stderr(self, clean=False):
        return self.get_dataarray('center_stderr', clean)

    @property
    def sigma(self, clean=True):
        return self.get_dataarray('sigma', clean)

    @property
    def amplitude(self, clean=True):
        return self.get_dataarray('amplitude', clean)

    @property
    def indexes(self):
        return self._data.center.indexes

    @property
    def coords(self):
        return self._data.center.coords

    @property
    def dims(self):
        return self._data.center.dims


class MultifitBand(Band):
    """
    Convenience class that reimplements reading data out of a composite fit result
    """

    def get_dataarray(self, var_name, clean=True):
        full_var_name = self.label + var_name

        if 'stderr' in full_var_name:
            return self._data.T.map(param_stderr_getter(full_var_name.split('_stderr')[0]))

        return self._data.T.map(param_getter(full_var_name))


class VoigtBand(Band):
    @property
    def fit_cls(self):
        return arpes.fits.VoigtModel


class BackgroundBand(Band):
    @property
    def fit_cls(self):
        return arpes.fits.GaussianModel


class FermiEdgeBand(Band):
    @property
    def fit_cls(self):
        return arpes.fits.GStepBStandardModel


class AffineBackgroundBand(Band):
    @property
    def fit_cls(self):
        return arpes.fits.AffineBackgroundModel
