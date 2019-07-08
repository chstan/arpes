import xarray as xr

import arpes.xarray_extensions
from arpes.endstations import HemisphericalEndstation, FITSEndstation


__all__ = ('ALGMainChamber',)

class ALGMainChamber(HemisphericalEndstation, FITSEndstation):
    PRINCIPAL_NAME = 'ALG-Main'
    ALIASES = ['MC', 'ALG-Main', 'ALG-MC', 'ALG-Hemisphere', 'ALG-Main Chamber',]

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        data = super().postprocess_final(data, scan_desc)
        for spectrum in data.S.spectra:
            spectrum.attrs['hv'] = 5.93 # only photon energy available on this chamber

        return data