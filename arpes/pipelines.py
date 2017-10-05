import xarray as xr

from arpes.corrections import apply_photon_energy_fermi_edge_correction, apply_quadratic_fermi_edge_correction
from arpes.preparation import dim_normalizer
from arpes.utilities import conversion
from .pipeline import pipeline, compose


@pipeline()
def correct_e_fermi_hv(arr: xr.DataArray):
    if isinstance(arr, xr.Dataset):
        arr = arr.raw

    if 'hv' not in arr.dims:
        return arr

    return apply_photon_energy_fermi_edge_correction(arr)


@pipeline()
def correct_e_fermi_spectrometer(arr: xr.DataArray):
    if 'phi' not in arr.dims:
        return arr

    return apply_quadratic_fermi_edge_correction(arr)


# Pipelines should never include data loading
# Scans are already normalized at this point, they should be whenever they are first
# interned in the netCDF format
convert_scan_to_kspace = compose(
    #remove_dead_pixels, TODO implement
    #lucy_richardson_deconvolution, TODO implement
    #trapezoid_correction, TODO implement, consider order
    pipeline('normalize_hv_axis')(dim_normalizer('hv')),
    correct_e_fermi_hv,
    correct_e_fermi_spectrometer,
    pipeline()(conversion.convert_to_kspace),
)

