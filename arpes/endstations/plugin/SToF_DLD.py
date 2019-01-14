import warnings
import copy
import os.path

import h5py
import numpy as np
import xarray as xr

import arpes.config


from arpes.endstations import EndstationBase
from arpes.provenance import provenance_from_file

__all__ = ('SToFDLDEndstation',)

class SToFDLDEndstation(EndstationBase):
    PRINCIPAL_NAME = 'ALG-SToF-DLD'

    def load(self, scan_desc: dict=None, **kwargs):
        """
        Imports a FITS file that contains all of the information from a run of Ping
        and Anton's delay line detector ARToF

        :param scan_desc: Dictionary with extra information to attach to the xarray.Dataset, must contain the location
        of the file
        :return: xarray.Dataset
        """

        if scan_desc is None:
            warnings.warn('Attempting to make due without user associated metadata for the file')
            raise TypeError('Expected a dictionary of metadata with the location of the file')

        metadata = copy.deepcopy(scan_desc)

        data_loc = metadata['file']
        data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

        f = h5py.File(data_loc, 'r')

        dataset_contents = dict()
        raw_data = f['/PRIMARY/DATA'][:]
        raw_data = raw_data[:, ::-1]  # Reverse the timing axis
        dataset_contents['raw'] = xr.DataArray(
            raw_data,
            coords={'x_pixels': np.linspace(0, 511, 512),
                    't_pixels': np.linspace(0, 511, 512)},
            dims=('x_pixels', 't_pixels'),
            attrs=f['/PRIMARY'].attrs.items(),
        )

        provenance_from_file(dataset_contents['raw'], data_loc, {
            'what': 'Loaded Anton and Ping DLD dataset from HDF5.',
            'by': 'load_DLD',
        })

        return xr.Dataset(
            dataset_contents,
            attrs=metadata
        )