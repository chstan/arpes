"""
Provides general utility methods that get used during the course of analysis.
"""

import copy
import itertools
import json
import os
import re
import warnings
from math import sin, cos
from operator import itemgetter

import datetime

from arpes.typing import DataType

import pandas as pd
import numpy as np
import xarray as xr

from arpes import constants
from .dataset import *
from .funcutils import *
from .normalize import *
from .region import *
from .attrs import *
from .collections import *


def enumerate_dataarray(arr: xr.DataArray):
    for coordinate in itertools.product(*[arr.coords[d] for d in arr.dims]):
        zip_location = dict(zip(arr.dims, (float(f) for f in coordinate)))
        yield zip_location, arr.loc[zip_location].values.item()


def fix_burnt_pixels(spectrum):
    """
    In reality the analyzers cannot provide perfect images for us. One of the
    principle failure modes is that individual pixels can get burnt out and will
    not provide any counts, or will provide consistently fewer or more than other
    pixels.

    Our approach here is to look for peaks in the difference across pixels and
    frames of a spectrum as indication of issues to be fixed. To patch the
    pixels, we replace them with the average value of their neighbors.

    spectrum - <npArray> containing the pixels

    returns: <npArray> containing the fixed pixels
    """
    pass


def jacobian_correction(energies, lattice_constant, theta, beta, alpha, phis, rhat):
    """
    Because converting from angles to momenta does not preserve area, we need
    to multiply by the Jacobian of the transformation in order to get the
    appropriate number of counts in the new cells.

    This differs across all the cells of a spectrum, because E and phi change.
    This function builds an array with the same shape that has the appropriate
    correction for each cell.

    energies - <npArray> the linear sampling of energies across the spectrum
    phis - <npArray> the linear sampling of angles across the spectrum

    returns: <npArray> a 2D array of the Jacobian correction to apply to each
    pixel in the spectrum
    """

    k0s = constants.K_INV_ANGSTROM * np.sqrt(energies) * lattice_constant / np.pi

    dkxdphi = (cos(theta) * cos(alpha) * np.cos(phis) -
               sin(theta) * np.sin(phis))

    dkydphi = (
        -cos(theta) * sin(beta) * np.sin(phis) +
        np.cos(phis) * (
            cos(beta) * sin(alpha) -
            cos(alpha) * sin(theta) * sin(beta)))

    # return the dot product
    rhat_x, rhat_y = rhat

    geometric_correction = rhat_x * dkxdphi + rhat_y * dkydphi
    return np.outer(k0s, geometric_correction)


def arrange_by_indices(items, indices):
    """
    This function is best illustrated by the example below. It arranges the
    items in the input according to the new indices that each item should occupy.

    It also has an inverse available in 'unarrange_by_indices'.

    Ex:
    arrange_by_indices(['a', 'b', 'c'], [1, 2, 0])
     => ['b', 'c', 'a']
    """
    return [items[i] for i in indices]


def unarrange_by_indices(items, indices):
    """
    The inverse function to 'arrange_by_indices'.

    Ex:
    unarrange_by_indices(['b', 'c', 'a'], [1, 2, 0])
     => ['a', 'b', 'c']
    """

    return [x for x, _ in sorted(zip(indices, items), key=itemgetter(0))]


def apply_dataarray(arr: xr.DataArray, f, *args, **kwargs):
    return xr.DataArray(
        f(arr.values, *args, **kwargs),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )


def lift_dataarray(f):
    """
    Lifts a function that operates on an np.ndarray's values to one that
    acts on the values of an xr.DataArray
    :param f:
    :return: g: Function operating on an xr.DataArray
    """

    def g(arr: xr.DataArray, *args, **kwargs):
        return apply_dataarray(arr, f, *args, **kwargs)

    return g


def lift_dataarray_attrs(f):
    """
    Lifts a function that operates on a dictionary to a function that acts on the
    attributes of an xr.DataArray, producing a new xr.DataArray. Another option
    if you don't need to create a new DataArray is to modify the attributes.
    :param f:
    :return: g: Function operating on the attributes of an xr.DataArray
    """

    def g(arr: xr.DataArray, *args, **kwargs):
        return xr.DataArray(
            arr.values,
            arr.coords,
            arr.dims,
            attrs=f(arr.attrs, *args, **kwargs)
        )

    return g


def lift_datavar_attrs(f):
    """
    Lifts a function that operates on a dictionary to a function that acts on the
    attributes of all the datavars in a xr.Dataset, as well as the Dataset attrs
    themselves.
    :param f: Function to apply
    :return:
    """

    def g(data: DataType, *args, **kwargs):
        arr_lifted = lift_dataarray_attrs(f)
        if isinstance(data, xr.DataArray):
            return arr_lifted(data, *args, **kwargs)

        var_names = list(data.data_vars.keys())
        new_vars = {k: arr_lifted(data[k], *args, **kwargs) for k in var_names}
        new_root_attrs = f(data.attrs, *args, **kwargs)

        return xr.Dataset(new_vars, data.coords, new_root_attrs)

    return g


def _rename_key(d, k, nk):
    if k in d:
        d[nk] = d[k]
        del d[k]


def rename_keys(d, keys_dict):
    d = d.copy()
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d


def clean_keys(d):
    def clean_single_key(k):
        k = k.replace(' ', '_')
        k = k.replace('.', '_')
        k = k.lower()
        k = re.sub(r'[()/?]', '', k)
        k = k.replace('__', '_')
        return k

    return dict(zip([clean_single_key(k) for k in d.keys()], d.values()))


rename_dataarray_attrs = lift_dataarray_attrs(rename_keys)
clean_attribute_names = lift_dataarray_attrs(clean_keys)

rename_datavar_attrs = lift_datavar_attrs(rename_keys)
clean_datavar_attribute_names = lift_datavar_attrs(clean_keys)

ATTRS_MAP = {
    'PuPol': 'pump_pol',
    'PrPol': 'probe_pol',
    'SFLNM0': 'lens_mode',
    'Lens Mode': 'lens_mode',
    'Excitation Energy': 'hv',
    'SFPE_0': 'pass_energy',
    'Pass Energy': 'pass_energy',
    'Slit Plate': 'slit',
    'Number of Sweeps': 'n_sweeps',
    'Acquisition Mode': 'scan_mode',
    'Region Name': 'scan_region',
    'Instrument': 'instrument',
    'Pressure': 'pressure',
    'User': 'user',
    'Polar': 'theta',
    'Theta': 'theta',
    'Sample': 'sample',
    'Beta': 'beta',
    'Azimuth': 'chi',
    'Location': 'location',
}

rename_standard_attrs = lambda x: rename_dataarray_attrs(x, ATTRS_MAP)
rename_datavar_standard_attrs = lambda x: rename_datavar_attrs(x, ATTRS_MAP)


def walk_scans(path, only_id=False):
    for path, _, files in os.walk(path):
        json_files = [f for f in files if '.json' in f]
        excel_files = [f for f in files if '.xlsx' in f or '.xlx' in f]

        for j in json_files:
            with open(os.path.join(path, j), 'r') as f:
                metadata = json.load(f)

            for scan in metadata:
                if only_id:
                    yield scan['id']
                else:
                    yield scan

        for x in excel_files:
            if 'cleaned' in x or 'cleaned' in path:
                continue

            ds = clean_xlsx_dataset(os.path.join(path, x))
            for file, scan in ds.iterrows():
                scan['file'] = scan.get('path', file)
                scan['short_file'] = file
                if only_id:
                    yield scan['id']
                else:
                    yield scan


def case_insensitive_get(d: dict, key: str, default=None, take_first=False):
    """
    Looks up a key in a dictionary ignoring case. We use this sometimes to be
    nicer to users who don't provide perfectly sanitized data
    :param d:
    :param key:
    :param default:
    :param take_first:
    :return:
    """
    found_value = False
    value = None

    for k, v in d.items():
        if k.lower() == key.lower():
            if not take_first and found_value:
                raise ValueError('Duplicate case insensitive keys')

            value = v
            found_value = True

            if take_first:
                break

    if not found_value:
        return default

    return value


_wrappable = {'note', 'data_preparation', 'provenance', 'corrections', 'symmetry_points'}
WHITELIST_KEYS = {'scan_region', 'sample', 'scan_mode', 'id', 'scan_mode'}
FREEZE_PROPS = {'spectrum_type'}


def wrap_attrs_dict(attrs: dict, original_data: DataType = None) -> dict:
    freeze_extra = []
    attrs_copy = copy.deepcopy(attrs)
    for key in _wrappable:
        if key not in attrs_copy:
            continue

        attrs_copy[key] = json.dumps(attrs_copy[key])

    for prop in FREEZE_PROPS:
        if prop not in original_data.attrs:
            try:
                resolved = normalize_to_spectrum(original_data).S
                attrs_copy[prop] = getattr(resolved, prop)
            except AttributeError:
                warnings.warn('Unresolvable attribute: {}'.format(prop))

    for k, v in attrs_copy.items():
        if v is None:
            freeze_extra.append(k)
            attrs_copy[k] = json.dumps(v)
        elif isinstance(v, bool):
            attrs_copy[k] = 1 if v else 0
        elif isinstance(v, (pd.Timestamp, datetime.time,)):
            attrs_copy[k] = v.isoformat()
        elif isinstance(v, (datetime.datetime,)):
            attrs_copy[k] = str(v)
        elif not isinstance(v, (str, float, int,)):
            print('Be careful about type: {}'.format(type(v)))

    def clean_key(key: str):
        return key.replace('#', 'num_')

    attrs_copy = {k: v for k, v in attrs_copy.items() if len(k)}
    attrs_copy = {clean_key(k): v for k, v in attrs_copy.items()}

    attrs_copy['freeze_extra'] = json.dumps(freeze_extra)
    return attrs_copy


def unwrap_attrs_dict(attrs: dict) -> dict:
    attrs_copy = copy.deepcopy(attrs)
    freeze_extra = json.loads(attrs_copy.pop('freeze_extra', '[]'))

    for key in _wrappable:
        if key not in attrs_copy:
            continue

        try:
            attrs_copy[key] = json.loads(attrs_copy[key])
        except json.JSONDecodeError:
            pass

    for frozen_extra in freeze_extra:
        try:
            attrs_copy[frozen_extra] = json.loads(attrs_copy[frozen_extra])
        except json.JSONDecodeError:
            warnings.warn('Was unable to unfreeze attribute "{}"'.format(frozen_extra))

    return attrs_copy


wrap_datavar_attrs = lift_datavar_attrs(wrap_attrs_dict)
unwrap_datavar_attrs = lift_datavar_attrs(unwrap_attrs_dict)
wrap_dataarray_attrs = lift_dataarray_attrs(wrap_attrs_dict)
unwrap_dataarray_attrs = lift_dataarray_attrs(unwrap_attrs_dict)
