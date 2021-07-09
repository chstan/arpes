"""Provides general utility methods that get used during the course of analysis."""

import itertools
from operator import itemgetter

import xarray as xr

from .funcutils import *
from .normalize import *
from .xarray import *
from .region import *
from .dict import *
from .attrs import *
from .collections import *


def enumerate_dataarray(arr: xr.DataArray):
    """Iterates through each coordinate location on n dataarray. Should merge to xarray_extensions."""
    for coordinate in itertools.product(*[arr.coords[d] for d in arr.dims]):
        zip_location = dict(zip(arr.dims, (float(f) for f in coordinate)))
        yield zip_location, arr.loc[zip_location].values.item()


def arrange_by_indices(items, indices):
    """Arranges `items` according to the new `indices` that each item should occupy.

    This function is best illustrated by the example below.
    It also has an inverse available in 'unarrange_by_indices'.

    Example:
        >>> arrange_by_indices(['a', 'b', 'c'], [1, 2, 0])
        ['b', 'c', 'a']
    """
    return [items[i] for i in indices]


def unarrange_by_indices(items, indices):
    """The inverse function to 'arrange_by_indices'.

    Ex:
    unarrange_by_indices(['b', 'c', 'a'], [1, 2, 0])
     => ['a', 'b', 'c']
    """
    return [x for x, _ in sorted(zip(indices, items), key=itemgetter(0))]


ATTRS_MAP = {
    "PuPol": "pump_pol",
    "PrPol": "probe_pol",
    "SFLNM0": "lens_mode",
    "Lens Mode": "lens_mode",
    "Excitation Energy": "hv",
    "SFPE_0": "pass_energy",
    "Pass Energy": "pass_energy",
    "Slit Plate": "slit",
    "Number of Sweeps": "n_sweeps",
    "Acquisition Mode": "scan_mode",
    "Region Name": "scan_region",
    "Instrument": "instrument",
    "Pressure": "pressure",
    "User": "user",
    "Polar": "theta",
    "Theta": "theta",
    "Sample": "sample",
    "Beta": "beta",
    "Azimuth": "chi",
    "Location": "location",
}
