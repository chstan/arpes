import numpy as np
import xarray as xr

from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, FITSEndstation

__all__ = ('MAESTROARPESEndstation',)


class MAESTROARPESEndstation(SynchrotronEndstation, HemisphericalEndstation, FITSEndstation):
    """
    The MERLIN ARPES Endstation at the Advanced Light Source
    """

    PRINCIPAL_NAME = 'ALS-BL702'
    ALIASES = ['BL7', 'BL7.0.2', 'ALS-BL7.0.2',]

    RENAME_KEYS = {
        'LMOTOR0': 'x',
        'LMOTOR1': 'y',
        'LMOTOR2': 'z',
        'Scan X': 'scan_x',
        'Scan Y': 'scan_y',
        'Scan Z': 'scan_z',
        'LMOTOR3': 'theta',
        'LMOTOR4': 'beta',
        'LMOTOR5': 'chi',
        'LMOTOR6': 'alpha',
        'LMOTOR9': 'psi',
        'mono_eV': 'hv',
        'SF_HV': 'hv',
        'SS_HV': 'hv',
        'Slit Defl': 'psi',

        # probably need something like an attribute list for extraction
        'SFRGN0': 'fixed_region_name',
        'SFE_0': 'daq_center_energy',
        'SFLNM0': 'lens_mode_name',
        'SFPE_0': 'pass_energy',
        'UNDHARM': 'undulator_harmonic',
        'RINGCURR': 'beam_current',
        'SFFR_0': 'frames_per_slice',

        'SFBA_0': 'phi_prebinning',
        'SFBE0': 'eV_prebinning',
    }

    RENAME_COORDS = {
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
    }

    ATTR_TRANSFORMS = {
        'START_T': lambda l: {'time': ' '.join(l.split(' ')[1:]).lower(),
                              'date': l.split(' ')[0]},
        'SF_SLITN': lambda l: {'slit_number': int(l.split(' ')[0]),
                               'slit_shape': l.split(' ')[-1].lower(),
                               'slit_width': float(l.split(' ')[2])},
    }

    MERGE_ATTRS = {
        'mcp_voltage': None,
        'repetition_rate': 5e8,
        'undulator_type': 'elliptically_polarized_undulator',
        'undulator_gap': None,
        'undulator_z': None,
        'undulator_polarization': None,
    }

    def load(self, scan_desc: dict = None, **kwargs):
        # in the future, can use a regex in order to handle the case where we postfix coordinates
        # for multiple spectra

        scan = super().load(scan_desc, **kwargs)

        coord_names = scan.coords.keys()
        will_rename = {}
        for coord_name in coord_names:
            if coord_name in self.RENAME_KEYS:
                will_rename[coord_name] = self.RENAME_KEYS.get(coord_name)

        renamed = scan.rename(will_rename)

        if 'scan_x' in renamed.coords:
            for d in renamed.data_vars:
                if 'spectrum' in d:
                    renamed[d].values = np.flip(renamed[d].values, axis=renamed[d].dims.index('scan_x'))

        return renamed

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        chamber = 'uARPES' if data.attrs['HOST2'] == 'uARPES.als.lbl.gov' else 'nARPES'
        if chamber == 'uARPES':
            analyzer_info = {
                'analyzer': 'R4000',
                'analyzer_name': 'Scienta R4000',
                'parallel_deflectors': False,
                'perpendicular_deflectors': True,
                'analyzer_radius': None,
                'analyzer_type': 'hemispherical',
            }
        else:
            analyzer_info = {
                'analyzer': 'DA-30',
                'analyzer_name': 'Scienta DA-30',
                'parallel_deflectors': False,
                'perpendicular_deflectors': False,
                'analyzer_radius': None,
                'analyzer_type': 'hemispherical',
            }

        ls = [data] + data.S.spectra
        for l in ls:
            l.attrs.update(analyzer_info)

            if 'GRATING' in l.attrs:
                l.attrs['grating_lines_per_mm'] = {
                    'G201b': 600,
                }.get(l.attrs['GRATING'])

        data = super().postprocess_final(data, scan_desc)

        return data
