import functools
import warnings

import numpy as np
import xarray as xr

from arpes.typing import DataType
from arpes.provenance import provenance
from arpes.utilities import normalize_to_spectrum

__all__ = ('curvature', 'dn_along_axis', 'd2_along_axis', 'd1_along_axis', 'minimum_gradient', 'vector_diff')


def vector_diff(arr, delta, n=1):
    """
    Computes finite differences along the vector delta, given as a tuple

    Using delta = (0, 1) is equivalent to np.diff(..., axis=1), while
    using delta = (1, 0) is equivalent to np.diff(..., axis=0).
    :param arr: np.ndarray
    :param delta: iterable containing vector to take difference along
    :return:
    """

    if n == 0:
        return arr
    if n < 0:
        raise ValueError(
            'Order must be non-negative but got ' + repr(n))

    nd = arr.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd

    for dim, delta_val in enumerate(delta):
        if delta_val != 0:
            if delta_val < 0:
                slice2[dim] = slice(-delta_val, None)
                slice1[dim] = slice(None, delta_val)
            else:
                slice1[dim] = slice(delta_val, None)
                slice2[dim] = slice(None, -delta_val)

    if n > 1:
        return vector_diff(arr[slice1] - arr[slice2], delta, n - 1)

    return arr[slice1] - arr[slice2]


def minimum_gradient(data: DataType,delta=1):
    arr = normalize_to_spectrum(data)
    new = arr / gradient_modulus(arr,delta=delta)
    new.values[np.isnan(new.values)] = 0
    return new

def gradient_modulus(data: DataType, delta=1):
    spectrum = normalize_to_spectrum(data)
    values = spectrum.values
    gradient_vector = np.zeros(shape=(8,) + values.shape)

    gradient_vector[0,:-delta,:] = vector_diff(values, (delta, 0,))
    gradient_vector[1,:,:-delta] = vector_diff(values, (0, delta,))
    gradient_vector[2,delta:,:] = vector_diff(values, (-delta, 0,))
    gradient_vector[3,:,delta:] = vector_diff(values, (0, -delta,))
    gradient_vector[4,:-delta,:-delta] = vector_diff(values, (delta, delta,))
    gradient_vector[5,:-delta,delta:] = vector_diff(values, (delta, -delta,))
    gradient_vector[6,delta:,:-delta] = vector_diff(values, (-delta, delta,))
    gradient_vector[7,delta:,delta:] = vector_diff(values, (-delta, -delta,))

    data_copy = spectrum.copy(deep=True)
    data_copy.values = np.linalg.norm(gradient_vector, axis=0)
    return data_copy


def curvature(arr: xr.DataArray, directions=None, alpha=1, beta=None):
    """
    Defined via
        C(x,y) = ([C_0 + (df/dx)^2]d^2f/dy^2 - 2 * df/dx df/dy d^2f/dxdy + [C_0 + (df/dy)^2]d^2f/dx^2) /
                 (C_0 (df/dx)^2 + (df/dy)^2)^(3/2)

    of in the case of inequivalent dimensions x and y

        C(x,y) = ([1 + C_x(df/dx)^2]C_y * d^2f/dy^2 -
                  2 * C_x * C_y * df/dx df/dy d^2f/dxdy +
                  [1 + C_y * (df/dy)^2] * C_x * d^2f/dx^2) /
                 (1 + C_x (df/dx)^2 + C_y (df/dy)^2)^(3/2)

        where
        C_x = C_y * (xi / eta)^2
        and where (xi / eta) = dx / dy

        The value of C_y can reasonably be taken to have the value |df/dx|_max^2 + |df/dy|_max^2
        C_y = (dy / dx) * (|df/dx|_max^2 + |df/dy|_max^2) * \alpha

        for some dimensionless parameter alpha
    :param arr:
    :param alpha: regulation parameter, chosen semi-universally, but with no particular justification
    :return:
    """
    if beta is not None:
        alpha = np.power(10., beta)

    if directions is None:
        directions = arr.dims[:2]

    axis_indices = tuple(arr.dims.index(d) for d in directions)
    dx, dy = tuple(float(arr.coords[d][1] - arr.coords[d][0]) for d in directions)
    dfx, dfy = np.gradient(arr.values, dx, dy, axis=axis_indices)
    np.nan_to_num(dfx, copy=False)
    np.nan_to_num(dfy, copy=False)

    mdfdx, mdfdy = np.max(np.abs(dfx)), np.max(np.abs(dfy))

    cy = (dy / dx) * (mdfdx ** 2 + mdfdy ** 2) * alpha
    cx = (dx / dy) * (mdfdx ** 2 + mdfdy ** 2) * alpha

    dfx_2, dfy_2 = np.power(dfx, 2), np.power(dfy, 2)
    d2fy = np.gradient(dfy, dy, axis=axis_indices[1])
    d2fx = np.gradient(dfx, dx, axis=axis_indices[0])
    d2fxy = np.gradient(dfx, dy, axis=axis_indices[1])

    denom = np.power((1 + cx * dfx_2 + cy * dfy_2), 1.5)
    numerator = (1 + cx * dfx_2) * cy * d2fy - 2 * cx * cy * dfx * dfy * d2fxy + \
                (1 + cy * dfy_2) * cx * d2fx

    curv = xr.DataArray(
        numerator / denom,
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if 'id' in curv.attrs:
        del curv.attrs['id']
        provenance(curv, arr, {
            'what': 'Curvature',
            'by': 'curvature',
            'directions': directions,
            'alpha': alpha,
        })
    return curv


def dn_along_axis(arr: xr.DataArray, axis=None, smooth_fn=None, order=2):
    """
    Like curvature, performs a second derivative. You can pass a function to use for smoothing through
    the parameter smooth_fn, otherwise no smoothing will be performed.

    You can specify the axis to take the derivative along with the axis param, which expects a string.
    If no axis is provided the axis will be chosen from among the available ones according to the preference
    for axes here, the first available being taken:

    ['eV', 'kp', 'kx', 'kz', 'ky', 'phi', 'beta', 'theta]
    :param arr:
    :param axis:
    :param smooth_fn:
    :param order: Specifies how many derivatives to take
    :return:
    """
    axis_order = ['eV', 'kp', 'kx', 'kz', 'ky', 'phi', 'beta', 'theta']
    if axis is None:
        axes = [a for a in axis_order if a in arr.dims]
        if len(axes):
            axis = axes[0]
        else:
            # have to do something
            axis = arr.dims[0]
            warnings.warn('Choosing axis: {} for the second derivative, no preferred axis found.'.format(axis))

    if smooth_fn is None:
        smooth_fn = lambda x: x

    d_axis = float(arr.coords[axis][1] - arr.coords[axis][0])
    axis_idx = arr.dims.index(axis)

    values = arr.values
    for _ in range(order):
        as_arr = xr.DataArray(values, arr.coords, arr.dims)
        smoothed = smooth_fn(as_arr)
        values = np.gradient(smoothed.values, d_axis, axis=axis_idx)

    dn_arr = xr.DataArray(
        values,
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if 'id' in dn_arr.attrs:
        del dn_arr.attrs['id']
        provenance(dn_arr, arr, {
            'what': '{}th derivative'.format(order),
            'by': 'dn_along_axis',
            'axis': axis,
            'order': order,
        })

    return dn_arr


d2_along_axis = functools.partial(dn_along_axis, order=2)
d1_along_axis = functools.partial(dn_along_axis, order=1)
