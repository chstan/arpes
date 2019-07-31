import warnings
import copy
import itertools
import os.path

import numpy as np
import xarray as xr

from astropy.io import fits

import arpes.config
from arpes.endstations import SynchrotronEndstation, HemisphericalEndstation
from arpes.endstations import find_clean_coords
from arpes.provenance import provenance_from_file
from arpes.utilities import rename_keys

__all__ = ('HERSEndstation',)

class HERSEndstation(SynchrotronEndstation, HemisphericalEndstation):
    """
    This should be unified with the FITs endstation code, but I don't have any projects at BL10
    at the moment so I will defer the complexity of unifying them for now
    """

    PRINCIPAL_NAME = 'ALS-BL1001'
    ALIASES = ['ALS-BL1001', 'HERS', 'ALS-HERS', 'BL1001']

    def load(self, scan_desc: dict=None, **kwargs):
        if scan_desc is None:
            warnings.warn('Attempting to make due without user associated scan_desc for the file')
            raise TypeError('Expected a dictionary of scan_desc with the location of the file')

        scan_desc = dict(copy.deepcopy(scan_desc))

        data_loc = scan_desc.get('path', scan_desc.get('file'))
        data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

        hdulist = fits.open(data_loc)

        hdulist[0].verify('fix+warn')
        header_hdu, hdu = hdulist[0], hdulist[1]

        coords, dimensions, spectrum_shape = find_clean_coords(hdu, scan_desc)
        columns = hdu.columns

        column_renamings = {}
        take_columns = columns

        spectra_names = [name for name in take_columns if name in columns.names]

        skip_frags = {}
        skip_predicates = {lambda k: any(s in k for s in skip_frags)}
        scan_desc = {k: v for k, v in scan_desc.items()
                    if not any(pred(k) for pred in skip_predicates)}

        data_vars = {k: (dimensions[k], hdu.data[k].reshape(spectrum_shape[k]), scan_desc)
                     for k in spectra_names}
        data_vars = rename_keys(data_vars, column_renamings)

        hdulist.close()

        relevant_dimensions = {k for k in coords.keys() if k in
                               set(itertools.chain(*[l[0] for l in data_vars.values()]))}
        relevant_coords = {k: v for k, v in coords.items() if k in relevant_dimensions}

        deg_to_rad_coords = {'beta', 'psi', 'chi', 'theta'}
        relevant_coords = {k: c * (np.pi / 180) if k in deg_to_rad_coords else c
                           for k, c in relevant_coords.items()}

        dataset = xr.Dataset(
            data_vars,
            relevant_coords,
            scan_desc,
        )

        provenance_from_file(dataset, data_loc, {
            'what': 'Loaded BL10 dataset',
            'by': 'load_DLD',
        })

        return dataset
