"""
Provides general utility methods that get used during the course of analysis.
A lot of these are borrowed/rewritten from other students and have very long
lineages.
"""

import functools
import itertools
import json
import math
import os
import re
import uuid
import warnings
from math import sin, cos, acos
from operator import itemgetter

import numpy as np
import pandas as pd
import xarray as xr

import arpes.constants
import arpes.materials


def enumerate_dataarray(arr: xr.DataArray):
    for coordinate in itertools.product(*[arr.coords[d] for d in arr.dims]):
        zip_location = dict(zip(arr.dims, (float(f) for f in coordinate)))
        yield zip_location, arr.loc[zip_location].values.item()

def split_hdu_header(value):
    """
    WARNING, this will remove all parentheses, so you might want to modify this or
    be more careful if you have a list of text data in your header!
    """
    return re.sub(r"[\(\)]", r"", value).split(',')

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


def denorm_lorentzian_with_background(x, background_level, amplitude, location, fwhm):
    """
    We should probably do proper background subtraction in some other way,
    but for now I'm being pretty rudimentary with things, so we're just including
    a background parameter for the Lorentzian itself
    """
    return background_level + amplitude/math.pi/(1 + ((x - location)/fwhm)**2)


def fermi_dirac_distribution(Es, mu, T):
    """
    Unitless Fermi-Dirac distribution
    This is meant to be fed into scipy.optimize or another fitting tool typically,
    or to generate data

    Es - <npArray> an array of all of the energy data

    returns: <npArray> the value of the distribution at different energies
    """

    return 1/(np.exp((Es-mu)/T) + 1)


def bose_einstein_distribution(Es, mu, T):
    return 1/(np.exp((Es-mu)/T) - 1)


def denorm_fermi_dirac_distribution(Es, mu, T, g):
    """
    Includes a constant 'density of states' g, this is mostly a convenience
    when working with scipy.optimize
    """
    return g*fermi_dirac_distribution(Es, mu, T)


def denorm_bose_einstein_distribution(Es, mu, T, g):
    """
    Includes a constant 'density of states' g, this is mostly a convenience
    when working with scipy.optimize
    """
    return g*bose_einstein_distribution(Es, mu, T)


def _prep_angles(angles, convert_radians=False):
    """
    Converts the analyzer angles to radians if required
    """
    return [(math.pi * a / 180) if convert_radians else a for a in angles]


# _rotation_proj_x through _rotation_proj_z are the only ones that depend on the
# geometry of the analyzer setup, so if you ever have to make changes,
# it's sufficient to check here
def _rotation_proj_x(theta, beta, alpha, phi):
    return (cos(theta) * cos(alpha) * sin(phi) +
            sin(theta) * cos(phi))


def _rotation_proj_y(theta, beta, alpha, phi):
    return (cos(theta) * sin(beta) * cos(phi) +
            sin(phi) * (
                cos(beta) * sin(alpha) -
                cos(alpha) * sin(theta) * sin(beta)))


def _rotation_proj_z(theta, beta, alpha, phi):
    return (cos(theta) * cos(beta) * cos(phi) -
            sin(phi) * (
                cos(alpha) * sin(theta) * cos(beta) +
                sin(beta) * sin(alpha)))


def _kk(angles, energy, lattice_constant, convert_radians=False, perform_rotation=None):
    """
    Converts the analyzer angles and the energy into a momentum.

    The default for the angles is in degrees, but you can pass radians if you specify
    not to convert the angles with the 'convert_radians' parameter.

    angles - <Tuple<Float>> The analyzer angles

    - alpha - the analyzer rotation angle
    - phi - the angle along the analyzer
    - beta and phi - the bellows and sample rotation angles as defined in the
      Spin-TOF experiment
    """

    k_inv_angstrom = 0.5123
    k0 = k_inv_angstrom * math.sqrt(energy) * lattice_constant / math.pi

    return k0 * perform_rotation(*_prep_angles(angles, convert_radians))

# The actual functions that we export to the world are the specializations
# of the angle conversion functions for each of x, y, and z
kkx = functools.partial(_kk, perform_rotation=_rotation_proj_x)
kky = functools.partial(_kk, perform_rotation=_rotation_proj_y)
kkz = functools.partial(_kk, perform_rotation=_rotation_proj_z)


def kkvec(*args, **kwargs):
    """
    Convenience function that returns the full three dimensional
    vector (kkx, kky, kkz,)

    See also 'kkxy' if you don't need all of the vector but just the x and y
    components
    """
    return (kkx(*args, **kwargs), kky(*args, **kwargs), kkz(*args, **kwargs),)


def kkxy(*args, **kwargs):
    """
    Convenience function that returns the two dimensional vector (kkx, kky,)
    """
    return (kkx(*args, **kwargs), kky(*args, **kwargs))


def angles_to_rhat(lattice_constant, energy, *angles):
    theta, beta, alpha, phi_min, phi_max = angles

    max_angles = (theta, beta, alpha, phi_max,)
    min_angles = (theta, beta, alpha, phi_min,)

    min_kkx, min_kky = kkxy((theta, beta, alpha, phi_min,), energy, lattice_constant)
    max_kkx, max_kky = kkxy((theta, beta, alpha, phi_max,), energy, lattice_constant)

    dkx, dky = max_kkx - min_kkx, max_kky - min_kky
    norm = math.sqrt(dkx * dkx + dky * dky)

    return (dkx / norm, dky / norm,)


def angles_to_k_dot_r(lattice_constant, theta, beta, alpha, phi, energy, rhat):
    angles = (theta, beta, alpha, phi,)
    theta, beta, alpha, phi = _prep_angles(angles, convert_radians=False)
    rhat_x, rhat_y = rhat
    x, y = kkxy((theta, beta, alpha, phi,), energy, lattice_constant)

    # return the dot product
    return rhat_x * x + rhat_y * y


def k_dot_r_to_angles(lattice_constant, theta, beta, alpha, k, energy, rhat):
    k_inv_angstrom = 0.5123
    k0 = k_inv_angstrom * math.sqrt(energy) * lattice_constant / math.pi

    rhat_x, rhat_y = rhat

    cos_component = k0 * (rhat_x * sin(theta) + rhat_y * cos(theta) * sin(beta))
    sin_component = k0 * (rhat_x * cos(theta) * cos(alpha) +
                          rhat_y * (cos(beta) * sin(alpha) -
                                    cos(alpha) * sin(beta) * sin(theta)))


    sign_phi = 1 if k - cos_component > 0 else -1
    perp_component = cos_component ** 2 + sin_component ** 2

    return sign_phi * acos(
        (cos_component * k + sin_component * math.sqrt(perp_component - k ** 2)) /
        perp_component)


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

    k_inv_angstrom = 0.5123
    k0s = k_inv_angstrom * np.sqrt(energies) * lattice_constant / math.pi

    dkxdphi = (cos(theta) * cos(alpha) * np.cos(phis) -
               sin(theta) * np.sin(phis))

    dkydphi = (
        -cos(theta) * sin(beta) * np.sin(phis) +
        np.cos(phis) * (
            cos(beta) * sin(alpha) -
            cos(alpha) * sin(theta) * sin(beta)))

    # return the dot product
    rhat_x, rhat_y = rhat

    geometric_correction = math.pi/180*(rhat_x * dkxdphi + rhat_y * dkydphi)
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

    return [x for x, _ in sorted(zip(indices, items), key=itemgetter[0])]


def get_spectrometer(dataset):
    spectrometers = {
        'SToF': arpes.constants.SPECTROMETER_SPIN_TOF,
        'ToF': arpes.constants.SPECTROMETER_STRAIGHT_TOF,
        'DLD': arpes.constants.SPECTROMETER_DLD,
    }

    if 'spectrometer_name' in dataset.attrs:
        return spectrometers.get(dataset.set.attrs['spectrometer_name'])

    if 'location' in dataset.attrs:
        return {
            'ALG-MC': arpes.constants.SPECTROMETER_MC,
            'BL403': arpes.constants.SPECTROMETER_BL4,
        }.get(dataset.attrs['location'])

    return spectrometers[dataset.attrs['spectrometer_name']]

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

def _rename_key(d, k, nk):
    if k in d:
        d[nk] = d[k]
        del d[k]

def rename_keys(d, keys_dict):
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d

def clean_keys(d):
    def clean_single_key(k):
        k = k.replace(' ', '_')
        return re.sub(r'[\(\)\/?]', '', k)

    return dict(zip([clean_single_key(k) for k in d.keys()], d.values()))


rename_dataarray_attrs = lift_dataarray_attrs(rename_keys)
clean_attribute_names = lift_dataarray_attrs(clean_keys)


rename_standard_attrs = lambda x: rename_dataarray_attrs(x, {
    'Lens Mode': 'lens_mode',
    'Excitation Energy': 'hv',
    'Pass Energy': 'pass_energy',
    'Slit Plate': 'slit',
    'Number of Sweepts': 'n_sweeps',
    'Acquisition Mode': 'scan_mode',
    'Region Name': 'scan_region',
    'Instrument': 'instrument',
    'Pressure': 'pressure',
    'User': 'user',
    'Polar': 'polar',
    'Sample': 'sample',
    'Beta': 'polar',
    'Location': 'location',
})


def clean_xlsx_dataset(path):
    base_filename, extension = os.path.splitext(path)
    if extension not in ['.xlsx', '.xlx']:
        warnings.warn('File is not an excel file')
        return None

    new_filename = base_filename + '.cleaned' + extension
    if os.path.exists(new_filename):
        return pd.read_excel(new_filename).set_index('file')

    ds = pd.read_excel(path, header=1, index_col=1)
    ds = ds.loc[ds.index.dropna()]

    last_index = None
    def is_blank(item):
        if isinstance(item, str):
            return item == ''

        if isinstance(item, float):
            return math.isnan(item)

        return False

    # Cascade blank values
    for index, row in ds.sort_index().iterrows():
        row = row.copy()

        for key, value in row.iteritems():
            if key == 'id' and math.isnan(float(row['id'])):
                ds.loc[index, ('id',)] = str(uuid.uuid1())

            elif last_index is not None and is_blank(value) and not is_blank(ds.loc[last_index,(key,)]):
                ds.loc[index,(key,)] = ds.loc[last_index,(key,)]

        last_index = index

    excel_writer = pd.ExcelWriter(new_filename)
    ds.to_excel(excel_writer)
    excel_writer.save()

    return ds

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



def polar_offset(arr: xr.DataArray):
    if 'polar_offset' in arr.attrs:
        return arr.attrs['polar_offset']

    return arr.attrs.get('data_preparation', {}).get('polar_offset', 0)


def phi_offset(arr: xr.DataArray):
    if 'polar_offset' in arr.attrs:
        return arr.attrs['phi_offset']

    return arr.attrs.get('data_preparation', {}).get('phi_offset', 0)


def material(arr: xr.DataArray):
    try:
        return arpes.materials.material_by_formula[arr.attrs['sample']]
    except:
        return None


def work_function(arr: xr.DataArray):
    if 'sample_workfunction' in arr.attrs:
        return arr.attrs['sample_workfunction']

    if material(arr):
        return material(arr).get('work_function', 4.32)

    return 4.32


def inner_potential(arr: xr.DataArray):
    if 'inner_potential' in arr.attrs:
        return arr.attrs['inner_potential']

    if material(arr):
        return material(arr).get('inner_potential', 10)

    return 10

def photon_energy(arr: xr.DataArray):
    if 'hv' in arr.attrs:
        return float(arr.attrs['hv'])

    return None