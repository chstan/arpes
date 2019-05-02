import numpy as np
import lmfit as lf
import matplotlib.pyplot as plt
import xarray as xr

from arpes.fits import QuadraticModel, GStepBModel, LinearModel, broadcast_model
from arpes.provenance import provenance
from arpes.typing import DataType
from arpes.utilities.math import shift_by
from arpes.utilities import normalize_to_spectrum


def _exclude_from_set(excluded):
    def exclude(l):
        return list(set(l).difference(excluded))

    return exclude


exclude_hemisphere_axes = _exclude_from_set({'phi', 'eV'})
exclude_hv_axes = _exclude_from_set({'hv', 'eV'})


__all__ = ['install_fermi_edge_reference', 'build_quadratic_fermi_edge_correction',
           'build_photon_energy_fermi_edge_correction', 'apply_photon_energy_fermi_edge_correction',
           'apply_quadratic_fermi_edge_correction', 'apply_copper_fermi_edge_correction',
           'apply_direct_copper_fermi_edge_correction', 'build_direct_fermi_edge_correction',
           'apply_direct_fermi_edge_correction', 'find_e_fermi_linear_dos',]


def find_e_fermi_linear_dos(edc, guess=None, plot=False, ax=None):
    """
    Does a reasonable job of finding E_Fermi in-situ for graphene/graphite or other materials with a linear DOS near
    the chemical potential. You can provide an initial guess via guess, or one will be chosen half way through the EDC.

    The Fermi level is estimated as the location where the DoS crosses below an estimated background level
    :param edc:
    :param guess:
    :param plot:
    :return:
    """
    if guess is None:
        guess = edc.eV.values[len(edc.eV) // 2]

    edc = edc - np.percentile(edc.values, (20,))[0]
    mask = edc > np.percentile(edc.sel(eV=slice(None, guess)), 20)
    mod = LinearModel().guess_fit(edc[mask])

    chemical_potential = -mod.params['intercept'].value / mod.params['slope'].value

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        edc.plot(ax=ax)
        ax.axvline(chemical_potential, linestyle='--', color='red')
        ax.axvline(guess, linestyle='--', color='gray')

    return chemical_potential


def install_fermi_edge_reference(arr: xr.DataArray):
    # TODO add method to install and reference corrections by looking at dataset metadata
    return build_quadratic_fermi_edge_correction(arr, plot=True)

def apply_direct_copper_fermi_edge_correction(arr: DataType, copper_ref: DataType, *args, **kwargs):
    """
    Applies a *direct* fermi edge correction.
    :param arr:
    :param copper_ref:
    :param args:
    :param kwargs:
    :return:
    """
    arr = normalize_to_spectrum(arr)
    copper_ref = normalize_to_spectrum(copper_ref)
    direct_corr = build_direct_fermi_edge_correction(copper_ref, *args, **kwargs)
    shift = np.interp(arr.coords['phi'].values, direct_corr.coords['phi'].values,
                      direct_corr.values)
    return apply_direct_fermi_edge_correction(arr, shift)

def apply_direct_fermi_edge_correction(arr: xr.DataArray, correction=None, *args, **kwargs):
    if correction is None:
        correction = build_direct_fermi_edge_correction(arr, *args, **kwargs)

    shift_amount = -correction / arr.T.stride(generic_dim_names=False)['eV']
    energy_axis = list(arr.dims).index('eV')

    correction_axis = list(arr.dims).index(correction.dims[0])

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=correction_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if 'id'in corrected_arr.attrs:
        del corrected_arr.attrs['id']

    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0 along hv axis',
        'by': 'apply_photon_energy_fermi_edge_correction',
        'correction': list(correction.values if isinstance(correction, xr.DataArray) else correction),
    })

    return corrected_arr

def build_direct_fermi_edge_correction(arr: xr.DataArray, fit_limit=0.001, energy_range=None, plot=False,
                                       along='phi'):
    """
    Builds a direct fermi edge correction stencil.

    This means that fits are performed at each value of the 'phi' coordinate
    to get a list of fits. Bad fits are thrown out to form a stencil.

    This can be used to shift coordinates by the nearest value in the stencil.

    :param copper_ref:
    :param args:
    :param kwargs:
    :return:
    """

    if energy_range is None:
        energy_range = slice(-0.1, 0.1)

    exclude_axes = ['eV', along]
    others = [d for d in arr.dims if d not in exclude_axes]
    edge_fit = broadcast_model(GStepBModel, arr.sum(others).sel(eV=energy_range), along).results

    def sieve(c, v):
        return v.item().params['center'].stderr < 0.001

    corrections = edge_fit.T.filter_coord(along, sieve).T.map(lambda x: x.params['center'].value)

    if plot:
        corrections.plot()

    return corrections


def apply_copper_fermi_edge_correction(arr: DataType, copper_ref: DataType, *args, **kwargs):
    # this maybe isn't best because we don't correct anything other than the spectrum,
    # but that's the only thing with an energy axis in ARPES datasets so whatever
    arr = normalize_to_spectrum(arr)
    copper_ref = normalize_to_spectrum(copper_ref)
    quadratic_corr = build_quadratic_fermi_edge_correction(copper_ref, *args, **kwargs)
    return apply_quadratic_fermi_edge_correction(arr, quadratic_corr)


def build_quadratic_fermi_edge_correction(arr: xr.DataArray, fit_limit=0.001, eV_slice=None, plot=False) -> lf.model.ModelResult:
    # TODO improve robustness here by allowing passing in the location of the fermi edge guess
    # We could also do this automatically by using the same method we use for step detection to find the edge of the
    # spectrometer image

    if eV_slice is None:
        approximate_fermi_level = arr.S.find_spectrum_energy_edges().max()
        eV_slice = slice(approximate_fermi_level-0.4, approximate_fermi_level+0.4)
    else:
        approximate_fermi_level = 0
    sum_axes = exclude_hemisphere_axes(arr.dims)
    edge_fit = broadcast_model(GStepBModel, arr.sum(sum_axes).sel(eV=eV_slice), 'phi', params={'center': {'value': approximate_fermi_level}})

    size_phi = len(arr.coords['phi'])
    not_nanny = (np.logical_not(np.isnan(arr)) * 1).sum('eV') > size_phi * 0.30
    condition = np.logical_and(edge_fit.F.s('center') < fit_limit, not_nanny)

    quadratic_corr = QuadraticModel().guess_fit(
        edge_fit.F.p('center'),
        weights=condition * 1)
    if plot:
        edge_fit.F.p('center').plot()
        plt.plot(arr.coords['phi'], quadratic_corr.best_fit)

    return quadratic_corr


def build_photon_energy_fermi_edge_correction(arr: xr.DataArray, plot=False, energy_window=0.2):
    edge_fit = broadcast_model(GStepBModel, arr.sum(exclude_hv_axes(arr.dims)).sel(
        eV=slice(-energy_window, energy_window)), 'hv')

    return edge_fit


def apply_photon_energy_fermi_edge_correction(arr: xr.DataArray, correction=None, **kwargs):
    if correction is None:
        correction = build_photon_energy_fermi_edge_correction(arr, **kwargs)

    correction_values = correction.T.map(lambda x: x.params['center'].value)
    if 'corrections' not in arr.attrs:
        arr.attrs['corrections'] = {}

    arr.attrs['corrections']['hv_correction'] = list(correction_values.values)

    shift_amount = -correction_values / arr.T.stride(generic_dim_names=False)['eV']
    energy_axis = arr.dims.index('eV')
    hv_axis = arr.dims.index('hv')

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=hv_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if 'id' in corrected_arr.attrs:
        del corrected_arr.attrs['id']

    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0 along hv axis',
        'by': 'apply_photon_energy_fermi_edge_correction',
        'correction': list(correction_values.values),
    })

    return corrected_arr

def apply_quadratic_fermi_edge_correction(arr: xr.DataArray, correction: lf.model.ModelResult=None, offset=None):
    assert(isinstance(arr, xr.DataArray))
    if correction is None:
        correction = build_quadratic_fermi_edge_correction(arr)

    if 'corrections' not in arr.attrs:
        arr.attrs['corrections'] = {}

    arr.attrs['corrections']['FE_Corr'] = correction.best_values

    delta_E = arr.coords['eV'].values[1] - arr.coords['eV'].values[0]
    dims = list(arr.dims)
    energy_axis = dims.index('eV')
    phi_axis = dims.index('phi')

    shift_amount_E = correction.eval(x=arr.coords['phi'].values)

    if offset is not None:
        shift_amount_E = shift_amount_E - offset

    shift_amount = -shift_amount_E / delta_E

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=phi_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    if 'id' in corrected_arr.attrs:
        del corrected_arr.attrs['id']

    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0',
        'by': 'apply_quadratic_fermi_edge_correction',
        'correction': correction.best_values,
    })

    return corrected_arr
