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
        'Polar Compens': 'theta', # these are caps-ed because they are dimensions in some cases!
        'BL Energy': 'hv',
        'tilt': 'beta', 'polar': 'theta', 'azimuth': 'chi',
        'temperature_sensor_a': 'temp_cryotip',
        'temperature_sensor_b': 'temp',
        'cryostat_temp_a': 'temp_cryotip',
        'cryostat_temp_b': 'temp',
        'bl_energy': 'hv',
        'polar_compens': 'theta',
        'K2200 V':'volts',
        'pwr_supply_v': 'volts'
    }

    def concatenate_frames(self, frames=typing.List[xr.Dataset], scan_desc: dict=None):
        if len(frames) < 2:
            return super().concatenate_frames(frames)

        # determine which axis to stitch them together along, and then do this
        original_filename = scan_desc.get('file', scan_desc.get('path'))
        assert(original_filename is not None)

        internal_match = re.match(r'([a-zA-Z0-9\w+_]+)_[S][0-9][0-9][0-9]\.pxt', Path(original_filename).name)
        if internal_match is not None:
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

                    for frame in frames:
                        # promote x, y, z to coords so they get concatted
                        for l in [frame] + frame.S.spectra:
                            for c in ['x', 'y', 'z']:
                                if c not in l.coords:
                                    l.coords[c] = l.attrs[c]

                    return xr.concat(frames, axis_name, coords='different')
                except Exception:
                    pass
        else:
            internal_match = re.match(r'([a-zA-Z0-9\w+_]+)_[R][0-9][0-9][0-9]\.pxt', Path(original_filename).name)
            if len(internal_match.groups()):
                return xr.merge(frames)

        return super().concatenate_frames(frames)

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        import copy
        import os
        from arpes.repair import negate_energy
        from arpes.load_pxt import read_single_pxt, find_ses_files_associated

        name, ext = os.path.splitext(frame_path)

        if 'nc' in ext:
            # was converted to hdf5/NetCDF format with Conrad's Igor scripts
            scan_desc = copy.deepcopy(scan_desc)
            scan_desc['path'] = frame_path
            return self.load_SES_nc(scan_desc=scan_desc, **kwargs)

        original_data_loc = scan_desc.get('path', scan_desc.get('file'))

        p = Path(original_data_loc)

        # find files with same name stem, indexed in format R###
        regions = find_ses_files_associated(p,separator='R')

        if len(regions) == 1:
            pxt_data = negate_energy(read_single_pxt(frame_path))
            return xr.Dataset({'spectrum': pxt_data}, attrs=pxt_data.attrs)
        else:
            # need to merge several different detector 'regions' in the same scan
            region_files = [self.load_single_region(region_path) for region_path in regions]

            # can they share their energy axes?
            all_same_energy = True
            for reg in region_files[1:]:
                dim = 'eV' + reg.attrs['Rnum']
                all_same_energy = all_same_energy and np.array_equal(region_files[0].coords['eV000'], reg.coords[dim])

            if all_same_energy:
                for i, reg in enumerate(region_files):
                    dim = 'eV' + reg.attrs['Rnum']
                    region_files[i] = reg.rename({dim: 'eV'})
            else:
                pass
            
            return self.concatenate_frames(region_files, scan_desc=scan_desc)

    def load_single_region(self, region_path: str = None, scan_desc: dict = None, **kwargs):
        import os
        from arpes.repair import negate_energy
        from arpes.load_pxt import read_single_pxt

        name, ext = os.path.splitext(region_path)
        num = name[-3:]

        pxt_data = negate_energy(read_single_pxt(region_path))
        pxt_data = pxt_data.rename({'eV': 'eV'+num})
        pxt_data.attrs['Rnum'] = num
        pxt_data.attrs['alpha'] = np.pi / 2
        return xr.Dataset({'spectrum' + num: pxt_data}, attrs=pxt_data.attrs)  # separate spectra for possibly unrelated data

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        ls = [data] + data.S.spectra
        deg_to_rad_coords = {'theta', 'phi', 'beta', 'chi', 'psi'}
        deg_to_rad_attrs = {'theta', 'beta', 'chi', 'psi', 'alpha'}

        for c in deg_to_rad_coords:
            if c in data.dims:
                data.coords[c] = data.coords[c] * np.pi / 180

        for angle_attr in deg_to_rad_attrs:
            for l in ls:
                if angle_attr in l.attrs:
                    l.attrs[angle_attr] = float(l.attrs[angle_attr]) * np.pi / 180

        data.attrs['alpha'] = np.pi / 2
        data.attrs['psi'] = 0
        for s in data.S.spectra:
            s.attrs['alpha'] = np.pi / 2
            s.attrs['psi'] = 0

        data = super().postprocess_final(data, scan_desc)

        return data
