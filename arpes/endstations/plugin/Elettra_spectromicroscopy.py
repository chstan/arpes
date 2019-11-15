import h5py
import numpy as np
import xarray as xr

from typing import Tuple

from arpes.endstations import (HemisphericalEndstation, SynchrotronEndstation, SingleFileEndstation)

__all__ = ('SpectromicroscopyElettraEndstation',)


def collect_coord(index: int, dset: h5py.Dataset) -> Tuple[str, np.ndarray]:
    """
    Uses the Spectromicroscopy beamline metadata format to normalize the coordinate information for a given axis.
    :param index:
    :param dset:
    :return:
    """
    shape = dset.shape
    name = dset.attrs[f'Dim{index} Name Units'][0].decode()
    start, delta = dset.attrs[f'Dim{index} Values']
    num = shape[index]
    coords = np.linspace(start, start + delta * (num - 1), num)
    return name, coords


def h5_dataset_to_dataarray(dset: h5py.Dataset) -> xr.DataArray:
    flat_coords = [collect_coord(i, dset) for i in range(len(dset.shape))]

    def unwrap_bytestring(possibly_bytestring):
        if isinstance(possibly_bytestring, bytes):
            return possibly_bytestring.decode()

        if isinstance(possibly_bytestring, (list, tuple, np.ndarray)):
            return [unwrap_bytestring(elem) for elem in possibly_bytestring]

        return possibly_bytestring

    DROP_KEYS = {
        'Dim0 Name Units',
        'Dim1 Name Units',
        'Dim2 Name Units',
        'Dim3 Name Units',
        'Dim0 Values',
        'Dim1 Values',
        'Dim2 Values',
        'Dim3 Values',
    }

    return xr.DataArray(
        dset[:],
        coords=dict(flat_coords),
        dims=[flat_coord[0] for flat_coord in flat_coords],
        attrs={k: unwrap_bytestring(v) for k, v in dset.attrs.items() if k not in DROP_KEYS},
    )


class SpectromicroscopyElettraEndstation(HemisphericalEndstation, SynchrotronEndstation, SingleFileEndstation):
    """
    Data loading for the nano-ARPES beamline "Spectromicroscopy Elettra".

    Information available on the beamline can be accessed at

    https://www.elettra.trieste.it/elettra-beamlines/spectromicroscopy.html.
    """

    PRINCIPAL_NAME = 'Spectromicroscopy Elettra'
    ALIASES = ['Spectromicroscopy', 'nano-ARPES Elettra']

    _TOLERATED_EXTENSIONS = {'.hdf5',}
    _SEARCH_PATTERNS = (
        r'[\-a-zA-Z0-9_\w]+_[0]+{}$',
        r'[\-a-zA-Z0-9_\w]+_{}$',
        r'[\-a-zA-Z0-9_\w]+{}$',
        r'[\-a-zA-Z0-9_\w]+[0]{}$',
    )

    ANALYZER_INFORMATION = {
        'analyzer': 'Custom: in vacuum hemispherical',
        'analyzer_name': 'Spectromicroscopy analyzer',
        'parallel_deflectors': False,
        'perpendicular_deflectors': False,
        'analyzer_radius': None,
        'analyzer_type': 'hemispherical',
    }

    RENAME_COORDS = {
        'KE': 'eV',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
    }

    RENAME_KEYS = {
        'Ep (eV)': 'pass_energy',
        'Dwell Time (s)': 'dwell_time',
        'Lens Mode': 'lens_mode',
        'MCP Voltage': 'mcp_voltage',
        'N of Scans': 'n_scans',
        'Pressure (mbar)': 'pressure',
        'Ring Current (mA)': 'ring_current',
        #'Ring En (GeV) Gap (mm) Photon(eV)': None,
        'Sample ID': 'sample',
        'Stage Coord (XYZR)': 'stage_coords',
        'Temperature (K)': 'temperature',
    }

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        with h5py.File(str(frame_path), 'r') as f:
            arrays = {k: h5_dataset_to_dataarray(f[k]) for k in f.keys()}

            if len(arrays) == 1:
                arrays = {'spectrum': list(arrays.values())[0]}

            return xr.Dataset(arrays)

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        data = data.rename({k: v for k, v in self.RENAME_COORDS.items() if k in data.coords.keys()})

        for i, dim_name in enumerate(['x', 'y', 'z']):
            if dim_name in data.coords:
                data.coords[dim_name] = data.coords[dim_name] / 1000.
            else:
                try:
                    data.coords[dim_name] = data.S.spectra[0].attrs['stage_coords'][i] / 1000.
                except:
                    data.coords[dim_name] = 0.

        data = super().postprocess_final(data, scan_desc)
        return data