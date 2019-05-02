from pathlib import Path
import xarray as xr
import numpy as np

from arpes.endstations import HemisphericalEndstation
from arpes.utilities import clean_keys

__all__ = ('MBSEndstation',)


class MBSEndstation(HemisphericalEndstation):
    """
    Implements loading text files from the MB Scientific text file format.

    There's not too much metadata here except what comes with the analyzer settings.
    """

    PRINCIPAL_NAME = 'MBS'
    ALIASES = ['MB Scientific',]

    RENAME_KEYS = {
    }

    def resolve_frame_locations(self, scan_desc: dict=None):
        return [scan_desc['path']]

    def load_single_frame(self, frame_path: str=None, scan_desc: dict=None, **kwargs):
        p = Path(frame_path)

        with open(str(p)) as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]
        data_index = lines.index('DATA:')
        header = lines[:data_index]
        data = lines[data_index + 1:]
        data = [d.split() for d in data]
        data = np.array([[float(f) for f in d] for d in data])

        header = [h.split('\t') for h in header]
        alt = [h for h in header if len(h) == 1]
        header = [h for h in header if len(h) == 2]
        header.append(['alt', str(alt)])
        attrs = clean_keys(dict(header))

        eV_axis = np.linspace(
            float(attrs['start_k_e_']),
            float(attrs['end_k_e_']),
            num=int(attrs['no_steps']),
            endpoint=False,
        )

        n_eV = int(attrs['no_steps'])
        idx_eV = data.shape.index(n_eV)

        if len(data.shape) == 2:
            phi_axis = np.linspace(
                float(attrs['xscalemin']),
                float(attrs['xscalemax']),
                num=data.shape[1 if idx_eV == 0 else 0],
                endpoint=False
            )

            coords = {'phi': phi_axis * np.pi / 180, 'eV': eV_axis}
            dims = ['eV', 'phi'] if idx_eV == 0 else ['phi', 'eV']
        else:
            coords = {'eV': eV_axis}
            dims = ['eV']


        return xr.Dataset({'spectrum': xr.DataArray(
            data, coords=coords, dims=dims, attrs=attrs
        )}, attrs=attrs)