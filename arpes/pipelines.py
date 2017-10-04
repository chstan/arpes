import xarray as xr

from .pipeline import pipeline, compose

from arpes.corrections import apply_photon_energy_fermi_edge_correction, apply_quadratic_fermi_edge_correction
from arpes.utilities.conversion import convert_to_kspace

@pipeline('correct_e_fermi_hv')
def correct_e_fermi_hv(arr: xr.DataArray):
    if 'hv' not in arr.dims:
        return arr

    return apply_photon_energy_fermi_edge_correction(arr)

@pipeline('correct_e_fermi_spectrometer')
def correct_e_fermi_spectrometer(arr: xr.DataArray):
    return apply_quadratic_fermi_edge_correction(arr)

@pipeline('convert_to_kspace')
def convert_to_kspace(arr: xr.DataArray):
    pass


# Pipelines should never include data loading
convert_scan_to_kspace = compose(
    #remove_dead_pixels, TODO implement
    correct_e_fermi_hv,
    correct_e_fermi_spectrometer,
    convert_to_kspace,
)

