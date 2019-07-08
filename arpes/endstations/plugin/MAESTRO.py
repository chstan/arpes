from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, FITSEndstation
import numpy as np

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
        'LMOTOR5': 'sample-phi',
        'LMOTOR9': 'polar',
        'mono_eV': 'hv',
        'SF_HV': 'hv',
        'SS_HV': 'hv',
        'Slit Defl': 'polar',
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