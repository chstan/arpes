import lmfit as lf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from arpes.fits import QuadraticModel, GStepBModel, broadcast_model
from arpes.provenance import provenance
from arpes.utilities import apply_dataarray
from arpes.utilities.math import shift_by


def _exclude_from_set(excluded):
    def exclude(l):
        return list(set(l).difference(excluded))

    return exclude


exclude_hemisphere_axes = _exclude_from_set({'phi', 'eV'})
exclude_hv_axes = _exclude_from_set({'hv', 'eV'})


__all__ = ['install_fermi_edge_reference', 'build_quadratic_fermi_edge_correction',
           'build_photon_energy_fermi_edge_correction', 'apply_photon_energy_fermi_edge_correction',
           'apply_quadratic_fermi_edge_correction']

def install_fermi_edge_reference(arr: xr.DataArray):
    # TODO add method to install and reference corrections by looking at dataset metadata
    return build_quadratic_fermi_edge_correction(arr, plot=True)


def build_quadratic_fermi_edge_correction(arr: xr.DataArray, fit_limit=0.001, plot=False) -> lf.model.ModelResult:
    # TODO improve robustness here by allowing passing in the location of the fermi edge guess
    # We could also do this automatically by using the same method we use for step detection to find the edge of the
    # spectrometer image
    edge_fit = broadcast_model(GStepBModel, arr.sum(exclude_hemisphere_axes(arr.dims)).sel(eV=slice(-0.1, 0.1)), 'phi')

    quadratic_corr = QuadraticModel().guess_fit(
        apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].value, otypes=[np.float])),
        weights=(apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].stderr, otypes=[np.float])).values
                 < fit_limit) * 1)
    if plot:
        apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].value)).plot()
        plt.plot(arr.coords['phi'], quadratic_corr.best_fit)

    return quadratic_corr


def build_photon_energy_fermi_edge_correction(arr: xr.DataArray, plot=False, energy_window=0.2):
    edge_fit = broadcast_model(GStepBModel, arr.sum(exclude_hv_axes(arr.dims)).sel(
        eV=slice(-energy_window, energy_window)), 'hv')

    return edge_fit


def apply_photon_energy_fermi_edge_correction(arr: xr.DataArray, correction=None, **kwargs):
    if correction is None:
        correction = build_photon_energy_fermi_edge_correction(arr, **kwargs)

    correction_values = apply_dataarray(correction, np.vectorize(lambda x: x.params['center'].value, otypes=[np.float]))
    if 'corrections' not in arr.attrs:
        arr.attrs['corrections'] = {}

    arr.attrs['corrections']['hv_correction'] = list(correction_values.values)

    shift_amount = -correction_values / (arr.coords['eV'].values[1] - arr.coords['eV'].values[0])
    energy_axis = arr.dims.index('eV')
    hv_axis = arr.dims.index('hv')

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=hv_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    del corrected_arr.attrs['id']
    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0 along hv axis',
        'by': 'apply_photon_energy_fermi_edge_correction',
        'correction': list(correction_values.values),
    })

    return corrected_arr

def apply_quadratic_fermi_edge_correction(arr: xr.DataArray, correction: lf.model.ModelResult=None):
    if correction is None:
        correction = build_quadratic_fermi_edge_correction(arr)

    if 'corrections' not in arr.attrs:
        arr.attrs['corrections'] = {}

    arr.attrs['corrections']['FE_Corr'] = correction.best_values

    delta_E = arr.coords['eV'].values[1] - arr.coords['eV'].values[0]
    energy_axis = arr.dims.index('eV')
    phi_axis = arr.dims.index('phi')
    shift_amount = -correction.best_fit / delta_E

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=phi_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    del corrected_arr.attrs['id']
    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0',
        'by': 'apply_quadratic_fermi_edge_correction',
        'correction': correction.best_values,
    })

    return corrected_arr
