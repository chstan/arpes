import typing
import numpy as np
import xarray as xr
import re
import pandas as pd
from pathlib import Path

from arpes.endstations import HemisphericalEndstation, SESEndstation

__all__ = ('KaindlEndstation',)

def find_kaindl_files_associated(reference_path: Path):
    name_match = re.match(r'([\w+]+_scan_[0-9][0-9][0-9]_)[0-9][0-9][0-9]\.pxt', reference_path.name)

    if name_match is None:
        return [reference_path]

    # otherwise need to collect all of the components
    fragment = name_match.groups()[0]
    components = list(reference_path.parent.glob('{}*.pxt'.format(fragment)))
    components.sort()

    return components

class KaindlEndstation(HemisphericalEndstation, SESEndstation):
    """
    The Robert Kaindl Endstation For HHG ARPES
    """

    PRINCIPAL_NAME = 'Kaindl'
    ALIASES = []

    RENAME_KEYS = {
        'Delay Stage': 'delay',
    }
    def resolve_frame_locations(self, scan_desc: dict=None):
        if scan_desc is None:
            raise ValueError('Must pass dictionary as file scan_desc to all endstation loading code.')

        original_data_loc = scan_desc.get('path', scan_desc.get('file'))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        p = Path(original_data_loc)
        return find_kaindl_files_associated(p)

    def concatenate_frames(self, frames=typing.List[xr.Dataset], scan_desc: dict=None):
        if len(frames) < 2:
            return super().concatenate_frames(frames)


        # determine which axis to stitch them together along, and then do this
        original_filename = scan_desc.get('path', scan_desc.get('file'))

        internal_match = re.match(r'([a-zA-Z0-9\w+_]+)_[0-9][0-9][0-9]\.pxt', Path(original_filename).name)
        if len(internal_match.groups()):
            motors_path = str(Path(original_filename).parent / '{}_Motor_Pos.txt'.format(internal_match.groups()[0]))
            AI_path = str(Path(original_filename).parent / '{}_AI.txt'.format(internal_match.groups()[0]))
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

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        original_filename = scan_desc.get('path', scan_desc.get('file'))
        internal_match = re.match(r'([a-zA-Z0-9\w+_]+_[0-9][0-9][0-9])\.pxt', Path(original_filename).name)
        all_filenames = find_kaindl_files_associated(Path(original_filename))
        all_filenames = ['{}\\{}_AI.txt'.format(f.parent,f.stem) for f in all_filenames]

        def load_attr_for_frame(filename, attr_name):
            df = pd.read_csv(filename, sep='\t', skiprows=6)
            return np.mean(df[attr_name])

        def attach_attr(data, attr_name, as_name):
            photocurrents = np.array([load_attr_for_frame(f, attr_name) 
                                      for f in all_filenames])
        
            if len(photocurrents) == 1:
                data[as_name] = photocurrents[0]
            else:
            
                non_spectrometer_dims = [d for d in data.spectrum.dims if d not in {'eV', 'phi'}]
                non_spectrometer_coords = {c: v for c, v in data.spectrum.coords.items() 
                                           if c in non_spectrometer_dims}

                new_shape = [len(data.coords[d]) for d in non_spectrometer_dims]
                photocurrent_arr = xr.DataArray(
                    photocurrents.reshape(new_shape), coords=non_spectrometer_coords, dims=non_spectrometer_dims)

                data = xr.merge([data, xr.Dataset(dict([[as_name, photocurrent_arr]]))])

            return data
        try:
            data = attach_attr(data, 'Photocurrent', 'photocurrent')
            data = attach_attr(data, 'Temperature B', 'temp')
            data = attach_attr(data, 'Temperature A', 'cryotip_temp')
        except FileNotFoundError as e:
            print(e)

        if len(internal_match.groups()):
            attrs_path = str(Path(original_filename).parent / '{}_AI.txt'.format(internal_match.groups()[0]))

            try:
                extra = pd.read_csv(attrs_path, sep='\t', skiprows=6)
                data = data.assign_attrs(extra=extra.to_json())
            except Exception:
                # WELP we tried
                pass

        deg_to_rad_coords = {'theta', 'beta', 'phi'}

        for c in deg_to_rad_coords:
            if c in data.dims:
                data.coords[c] = data.coords[c] * np.pi / 180

        deg_to_rad_attrs = {'theta', 'beta', 'alpha', 'chi'}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in data.attrs:
                data.attrs[angle_attr] = float(data.attrs[angle_attr]) * np.pi / 180

        data = super().postprocess_final(data, scan_desc)

        return data
