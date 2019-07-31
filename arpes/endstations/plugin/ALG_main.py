import xarray as xr
import numpy as np

import arpes.xarray_extensions
from arpes.endstations import HemisphericalEndstation, FITSEndstation


__all__ = ('ALGMainChamber',)


class ALGMainChamber(HemisphericalEndstation, FITSEndstation):
    PRINCIPAL_NAME = 'ALG-Main'
    ALIASES = ['MC', 'ALG-Main', 'ALG-MC', 'ALG-Hemisphere', 'ALG-Main Chamber',]

    RENAME_KEYS = {
        'Phi': 'chi',
        'Beta': 'beta',
        'Theta': 'theta',
        'Azimuth': 'chi',
        'Alpha': 'alpha',
        'Pump_energy_uJcm2': 'pump_fluence',
        'T0_ps': 't0_nominal',
        'W_func': 'workfunction',
        'Slit': 'slit',
        'LMOTOR0': 'x',
        'LMOTOR1': 'y',
        'LMOTOR2': 'z',
        'LMOTOR3': 'theta',
        'LMOTOR4': 'beta',
        'LMOTOR5': 'chi',
        'LMOTOR6': 'delay',
    }

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        data.attrs['hv'] = 5.93
        data.attrs['alpha'] = 0
        data.attrs['psi'] = 0

        # by default we use this value since this isnear the center of the spectrometer window
        data.attrs['phi_offset'] = 0.405
        for spectrum in data.S.spectra:
            spectrum.attrs['hv'] = 5.93  # only photon energy available on this chamber
            spectrum.attrs['alpha'] = 0
            spectrum.attrs['psi'] = 0
            spectrum.attrs['phi_offset'] = 0.405

        data = super().postprocess_final(data, scan_desc)

        if 'beta' in data.coords:
            data.coords['beta'].values = data.coords['beta'].values * np.pi / 180

        return data
