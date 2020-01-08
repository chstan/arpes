import os
import h5py
import numpy as np
import xarray as xr
from pathlib import Path
import typing

import arpes.config

from typing import Tuple

from arpes.endstations import (HemisphericalEndstation, SynchrotronEndstation)
from arpes.utilities import unwrap_xarray_item

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
    if name == 'P':
        name = 'phi'
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

    coords = dict(flat_coords)
    attrs = {k: unwrap_bytestring(v) for k, v in dset.attrs.items() if k not in DROP_KEYS}

    # attr normalization
    attrs['T'] = attrs['Angular Coord'][0]
    attrs['P'] = attrs['Angular Coord'][1]

    coords['P'] = attrs['P']

    del attrs['Angular Coord']  # temp
    del attrs['Date Time Start Stop']  # temp
    del attrs['Temperature (K)']  # temp
    del attrs['DET Limits']  # temp
    del attrs['Energy Window (eV)']  # temp
    del attrs['Ring En (GeV) GAP (mm) Photon (eV)']  # temp
    del attrs['Ring Current (mA)']  # temp
    del attrs['Stage Coord (XYZR)']  # temp

    return xr.DataArray(
        dset[:],
        coords=coords,
        dims=[flat_coord[0] for flat_coord in flat_coords],
        attrs=attrs,
    )


class SpectromicroscopyElettraEndstation(HemisphericalEndstation, SynchrotronEndstation):
    """
    Data loading for the nano-ARPES beamline "Spectromicroscopy Elettra".

    Information available on the beamline can be accessed
    `here <https://www.elettra.trieste.it/elettra-beamlines/spectromicroscopy>`_.
    """

    PRINCIPAL_NAME = 'Spectromicroscopy Elettra'
    ALIASES = ['Spectromicroscopy', 'nano-ARPES Elettra']

    _TOLERATED_EXTENSIONS = {'.hdf5',}
    _SEARCH_PATTERNS = (
        r'[\-a-zA-Z0-9_\w]+_[0]+{}$',
        r'[\-a-zA-Z0-9_\w]+_{}$',
        r'[\-a-zA-Z0-9_\w]+{}$',
        r'[\-a-zA-Z0-9_\w]+[0]{}$',
        r'{}' + (r'\\' if os.path.sep == '\\' else '/') + r'[\-a-zA-Z0-9_\w]+_001$',
    )

    @classmethod
    def files_for_search(cls, directory):
        base_files = []
        for file in os.listdir(directory):
            p = os.path.join(directory, file)
            if os.path.isdir(p):
                base_files = base_files + [os.path.join(file, f) for f in os.listdir(p)]
            else:
                base_files = base_files + [file]

        return list(filter(lambda f: os.path.splitext(f)[1] in cls._TOLERATED_EXTENSIONS, base_files))

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

    CONCAT_COORDS = ['T', 'P']

    def concatenate_frames(self, frames=typing.List[xr.Dataset], scan_desc: dict = None):
        if not frames:
            raise ValueError('Could not read any frames.')

        if len(frames) == 1:
            return frames[0]

        # determine which axis to stitch them together along, and then do this
        scan_coord = None
        max_different_values = -np.inf
        best_coordinates = []

        for possible_scan_coord in self.CONCAT_COORDS:
            coordinates = [f.coords.get(possible_scan_coord, None) for f in frames]
            coordinates = [None if hasattr(c, 'shape') and len(c.shape) else unwrap_xarray_item(c) for c in coordinates]

            n_different_values = len(set(coordinates))
            if n_different_values > max_different_values and None not in coordinates:
                max_different_values = n_different_values
                scan_coord = possible_scan_coord
                best_coordinates = coordinates

        assert scan_coord is not None

        fs = []
        for c, f in zip(best_coordinates, frames):
            f = f.spectrum
            f.coords[scan_coord] = c
            fs.append(f)

        return xr.Dataset({'spectrum': xr.concat(fs, scan_coord)})

    def resolve_frame_locations(self, scan_desc: dict = None):
        if scan_desc is None:
            raise ValueError('Must pass dictionary as file scan_desc to all endstation loading code.')

        original_data_loc = scan_desc.get('path', scan_desc.get('file'))

        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        p = Path(original_data_loc)

        if p.parent.parent.stem in (list(self._SEARCH_DIRECTORIES) + ['data']):
            return list(p.parent.glob('*.hdf5'))

        return [p]

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