import xarray as xr

from arpes.fits import GStepBModel, broadcast_model
from arpes.utilities.math import shift_by
from arpes.utilities.normalize import normalize_to_spectrum

from arpes.typing import DataType

__all__ = ('apply_cycle_fermi_edge_correction', 'build_cycle_fermi_edge_correction',)


def build_cycle_fermi_edge_correction(data: DataType, energy_range=None):
    arr = normalize_to_spectrum(data)

    if 'pixels' in arr.dims or 'phi' in arr.dims:
        arr = arr.S.region_sel('wide_angular')

    if energy_range is None:
        energy_range = slice(-0.1, 0.1)

    arr = arr.S.sum_other(['eV', 'cycle'])
    return broadcast_model(GStepBModel, arr.sel(eV=energy_range), 'cycle')


def apply_cycle_fermi_edge_correction(data: DataType, energy_range=None, shift=True):
    correction = build_cycle_fermi_edge_correction(data, energy_range)

    if shift:
        correction_values = correction.T.map(lambda x: x.params['center'].value)
        correction_shift = - correction_values / data.S.spectrum.T.stride(generic_dim_names=False)['eV']
        dataarr = data.S.spectrum
        shifted = xr.DataArray(
            shift_by(dataarr.values, correction_shift, axis=dataarr.dims.index('eV'),
                     by_axis=dataarr.dims.index('cycle'), order=1),
            dataarr.coords,
            dataarr.dims,
            attrs=dataarr.attrs.copy()
        )
        if 'id' in shifted.attrs:
            del shifted.attrs['id']

        return shifted
    else:
        raise NotImplementedError('Need to do l2 fit, then shift axis')
