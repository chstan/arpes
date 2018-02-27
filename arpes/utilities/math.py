import collections
import itertools

import xarray as xr

import numpy as np
import scipy.ndimage.interpolation

import arpes.constants


def derivative(f, arg_idx=0):
    """
    Defines a simple midpoint derivative
    """

    def d(*args):
        args = list(args)
        ref_arg = args[arg_idx]
        d = ref_arg / 100

        args[arg_idx] = ref_arg + d
        high = f(*args)
        args[arg_idx] = ref_arg - d
        low = f(*args)

        return (high - low) / (2 * d)

    return d


def polarization(up, down):
    return (up - down) / (up + down)


def propagate_statistical_error(f):
    def compute_propagated_error(*args):
        running_sum = 0
        for i, arg in enumerate(args):
            df_darg_i = derivative(f, i)
            running_sum += df_darg_i(*args) ** 2 * arg

        return np.sqrt(running_sum)

    return compute_propagated_error

def shift_by(arr, value, axis=0, by_axis=0, **kwargs):
    assert(axis != by_axis)
    arr_copy = arr.copy()

    if isinstance(value, xr.DataArray):
        value = value.values

    if not isinstance(value, collections.Iterable):
        value = list(itertools.repeat(value, times=arr.shape[by_axis]))

    for axis_idx in range(arr.shape[by_axis]):
        slc = (slice(None),) * by_axis + (axis_idx,) + (slice(None),) * (arr.ndim - by_axis - 1)
        shift_amount = (0,) * axis + (value[axis_idx],) + (0,) * (arr.ndim - axis - 1)

        if axis > by_axis:
            shift_amount = shift_amount[1:]
        else:
            shift_amount = shift_amount[:-1]

        arr_copy[slc] = scipy.ndimage.interpolation.shift(arr[slc], shift_amount, **kwargs)

    return arr_copy


def inv_fermi_distribution(energy, temperature, mu=0):
    """Expects energy in eV and temperature in Kelvin."""
    return np.exp((energy - mu) / (arpes.constants.K_BOLTZMANN_EV_KELVIN * temperature)) + 1


def fermi_distribution(energy, temperature):
    """Expects energy in eV and temperature in Kelvin."""
    return 1 / inv_fermi_distribution(energy, temperature)