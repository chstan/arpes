import typing
import numpy as np
import xarray as xr
import re
from pathlib import Path

from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, SESEndstation

__all__ = ('BL403ARPESEndstation',)


class BL403ARPESEndstation(SynchrotronEndstation, HemisphericalEndstation, SESEndstation):
    """
    The MERLIN ARPES Endstation at the Advanced Light Source
    """

    PRINCIPAL_NAME = 'ALS-BL403'
    ALIASES = ['BL403', 'BL4', 'BL4.0.3', 'ALS-BL403', 'ALS-BL4',]

    RENAME_KEYS = {
        'Tilt': 'theta', 'Polar': 'polar', 'Azimuth': 'chi',
        'Sample X': 'x', 'Sample Y (Vert)': 'y', 'Sample Z': 'z',
        'Temperature Sensor A': 'temp_cryotip',
        'temperature_sensor_a': 'temp_cryotip',
        'Temperature Sensor B': 'temp',
        'temperature_sensor_b': 'temp',
        'Cryostat Temp A': 'temp_cryotip',
        'Cryostat Temp B': 'temp',
        'BL Energy': 'hv',
    }

    def concatenate_frames(self, frames=typing.List[xr.Dataset], scan_desc: dict=None):
        if len(frames) < 2:
            return super().concatenate_frames(frames)


        # determine which axis to stitch them together along, and then do this
        original_filename = scan_desc['file']

        internal_match = re.match(r'([a-zA-Z0-9\w+_]+)_S[0-9][0-9][0-9]\.pxt', Path(original_filename).name)
        if len(internal_match.groups()):
            motors_path = str(Path(original_filename).parent / '{}_Motor_Pos.txt'.format(internal_match.groups()[0]))
            try:
                with open(motors_path, 'r') as f:
                    lines = f.readlines()

                axis_name = lines[0].strip()
                axis_name = self.RENAME_KEYS.get(axis_name, axis_name)
                values = [float(l.strip()) for l in lines[1:len(frames) + 1]]

                for v, f in zip(values, frames):
                    f.coords[axis_name] = v

                frames.sort(key=lambda x: x.coords[axis_name])
                return xr.concat(frames, axis_name)
            except Exception:
                pass

        return super().concatenate_frames(frames)

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        deg_to_rad_coords = {'polar', 'phi'}

        for c in deg_to_rad_coords:
            if c in data.dims:
                data.coords[c] = data.coords[c] * np.pi / 180

        deg_to_rad_attrs = {'theta', 'polar', 'chi'}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in data.attrs:
                data.attrs[angle_attr] = float(data.attrs[angle_attr]) * np.pi / 180

        data = super().postprocess_final(data, scan_desc)

        return data
