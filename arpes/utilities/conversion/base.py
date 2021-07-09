"""Infrastructure code for defining coordinate transforms and momentum conversion."""
import numpy as np

import xarray as xr
from typing import Any

__all__ = ["CoordinateConverter", "K_SPACE_BORDER", "MOMENTUM_BREAKPOINTS"]

K_SPACE_BORDER = 0.02
MOMENTUM_BREAKPOINTS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]


class CoordinateConverter:
    """Infrastructure code to support a new coordinate conversion routine.

    In order to do coordinate conversion from c_i to d_i, we need to give functions
    c_i(d_j), i.e. to implement the inverse transform. This is so that we convert by
    interpolating the function from a regular grid of d_i values back to the original
    data expressed in c_i.

    From this, we can see what responsibilities these conversion classes hold:

    * They need to specify how to calculate c_i(d_j)
    * They need to cache computations so that computations of c_i(d_j) can be performed
      efficiently for different coordinates c_i
    * Because they know how to do the inverse conversion, they need to know how to choose
      reasonable grid bounds for the forward transform, so that this can be handled
      automatically.

    These different roles and how they are accomplished are discussed in detail below.
    """

    def __init__(self, arr: xr.DataArray, dim_order=None, calibration=None, *args, **kwargs):
        """Intern the volume so that we can check on things during computation."""
        self.arr = arr
        self.dim_order = dim_order
        self.calibration = calibration

    def prep(self, arr: xr.DataArray):
        """Perform preprocessing of the array to convert before we start.

        The CoordinateConverter.prep method allows you to pre-compute some transformations
        that are common to the individual coordinate transform methods as an optimization.

        This is useful if you want the conversion methods to have separation of concern,
        but if it is advantageous for them to be able to share a computation of some
        variable. An example of this is in BE-kx-ky conversion, where computing k_p_tot
        is a step in both converting kx and ky, so we would like to avoid doing it twice.

        Of course, you can neglect this function entirely. Another technique is to simple
        cache computations as they arrive. This is the technique that is used in
        ConvertKxKy below
        """
        pass

    @property
    def is_slit_vertical(self) -> bool:
        """For hemispherical analyzers, whether the slit is vertical or horizontal.

        This is an ARPES specific detail, so this conversion code is not strictly general, but
        a future refactor could just push these details to a subclass.
        """
        # 89 - 91 degrees
        return np.abs(self.arr.S.lookup_offset_coord("alpha") - np.pi / 2) < (np.pi / 180)

    def kspace_to_BE(
        self, binding_energy: np.ndarray, *args: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """The energy conservation equation for ARPES.

        This does not depend on any details of the angular conversion (it's the identity) so we can
        put the conversion code here in the base class.
        """
        return binding_energy

    def conversion_for(self, dim):
        """Fetches the method responsible for calculating `dim` from momentum coordinates."""
        pass

    def identity_transform(self, axis_name, *args, **kwargs):
        """Just returns the coordinate requested from args.

        Useful if the transform is the identity.
        """
        return args[self.dim_order.index(axis_name)]

    def get_coordinates(self, resolution: dict = None, bounds: dict = None):
        """Calculates the coordinates which should be used in momentum space."""
        coordinates = {}
        coordinates["eV"] = self.arr.coords["eV"]
        return coordinates
