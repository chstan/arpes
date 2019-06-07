import xarray as xr
import warnings

import arpes.utilities

__all__ = ['CoordinateConverter', 'K_SPACE_BORDER', 'MOMENTUM_BREAKPOINTS']

K_SPACE_BORDER = 0.02
MOMENTUM_BREAKPOINTS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

class CoordinateConverter(object):
    def __init__(self, arr: xr.DataArray, dim_order=None, *args, **kwargs):
        # Intern the volume so that we can check on things during computation
        self.arr = arr
        self.dim_order = dim_order

        spectrometer = self.arr.S.spectrometer
        if spectrometer is not None:
            self.spectrometer = spectrometer
            self.is_slit_vertical = self.arr.S.is_slit_vertical
        else:
            warnings.warn('Could not find spectrometer information! Ask Conrad for help.')

    def prep(self, arr: xr.DataArray):
        """
        The CoordinateConverter.prep method allows you to pre-compute some transformations
        that are common to the individual coordinate transform methods as an optimization.

        This is useful if you want the conversion methods to have separation of concern,
        but if it is advantageous for them to be able to share a computation of some
        variable. An example of this is in BE-kx-ky conversion, where computing k_p_tot
        is a step in both converting kx and ky, so we would like to avoid doing it twice.

        Of course, you can neglect this function entirely. Another technique is to simple
        cache computations as they arrive. This is the technique that is used in
        ConvertKxKy below
        :param arr:
        :return:
        """
        pass

    def kspace_to_BE(self, binding_energy, *args, **kwargs):
        return binding_energy

    def conversion_for(self, dim):
        pass

    def identity_transform(self, axis_name, *args, **kwargs):
        return args[self.dim_order.index(axis_name)]

    def get_coordinates(self, resolution: dict=None):
        coordinates = {}
        coordinates['eV'] = self.arr.coords['eV']
        return coordinates