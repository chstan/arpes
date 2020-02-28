import xarray as xr
import numpy as np

from scipy.interpolate import make_lsq_spline

__all__ = ('simple_lsq_spline',)


def simple_lsq_spline(arr: xr.DataArray, order=3):
    xs = arr.coords[arr.dims[0]].values
    ys = arr.values

    knots = np.linspace(xs[0], xs[-1], order)
    t = np.r_[(xs[0],) * (order - 1), knots, (xs[-1],) * (order - 1)]

    spl = make_lsq_spline(xs, ys, t, order - 1)
    return spl