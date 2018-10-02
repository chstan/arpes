import warnings
import copy
import os.path
import xarray as xr
import numpy as np

import h5py

import arpes.config

from astropy.io import fits
from arpes.endstations import EndstationBase
from endstations import find_clean_coords
from provenance import provenance_from_file
from utilities import rename_keys

__all__ = ('SpinToFEndstation',)


class SpinToFEndstation(EndstationBase):
    PRINCIPAL_NAME = 'ALG-SToF'
    ALIASES = ['ALG-SToF', 'SToF', 'Spin-ToF', 'ALG-SpinToF']
    SKIP_ATTR_FRAGMENTS = {
        'MMX', 'TRVAL', 'TRDELT', 'COMMENT', 'OFFSET', 'SMOTOR', 'TUNIT', 'PMOTOR',
        'LMOTOR', 'TDESC', 'NAXIS', 'TTYPE', 'TFORM', 'XTENSION', 'BITPIX', 'TDELT',
        'TRPIX',
    }

    COLUMN_RENAMINGS = {
        'TempA': 'temperature_cryo',
        'TempB': 'temperature_sample',
        'Current': 'photocurrent',
        'ALS_Beam_mA': 'beam_current',
        'Energy_Spectra': 'spectrum',
        'targetPlus': 'up',
        'targetMinus': 'down',
        'wave': 'spectrum',  # this should not occur simultaneously with 'Energy_Spectra'
        'Time_Target_Up': 'up',
        'Time_Target_Down': 'down',
    }

    def load_SToF_hdf5(self, scan_desc: dict=None, **kwargs):
        """
        Imports a FITS file that contains ToF spectra.

        :param scan_desc: Dictionary with extra information to attach to the xr.Dataset, must contain the location
        of the file
        :return: xr.Dataset
        """

        scan_desc = copy.deepcopy(scan_desc)

        data_loc = scan_desc.get('path', scan_desc.get('file'))
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
            attrs=scan_desc
        )

    def load_SToF_fits(self, scan_desc: dict=None, **kwargs):
        scan_desc = dict(copy.deepcopy(scan_desc))

        data_loc = scan_desc.get('path', scan_desc.get('file'))
        data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

        hdulist = fits.open(data_loc)

        hdulist[0].verify('fix+warn')
        header_hdu, hdu = hdulist[0], hdulist[1]

        scan_desc.update(dict(hdu.header))
        scan_desc.update(dict(header_hdu.header))

        drop_attrs = ['COMMENT', 'HISTORY', 'EXTEND', 'SIMPLE', 'SCANPAR', 'SFKE_0']
        for dropped_attr in drop_attrs:
            if dropped_attr in scan_desc:
                del scan_desc[dropped_attr]

        coords, dimensions, spectrum_shape = find_clean_coords(hdu, scan_desc)

        columns = hdu.columns

        spin_column_names = {'targetMinus', 'targetPlus', 'Time_Target_Up', 'Time_Target_Down', 'Energy_Target_Up',
                             'Energy_Target_Down'}

        is_spin_resolved = any(cname in columns.names for cname in spin_column_names)
        spin_columns = ['Current' 'TempA', 'TempB', 'ALS_Beam_mA'] + list(spin_column_names)
        straight_columns = ['Current', 'TempA', 'TempB', 'ALS_Beam_mA', 'Energy_Spectra', 'wave']
        take_columns = spin_columns if is_spin_resolved else straight_columns

        # We could do our own spectrum conversion too, but that would be more annoying
        # it would slightly improve accuracy though
        spectra_names = [name for name in take_columns if name in columns.names]
        # }, column_renamings)

        skip_frags = {'MMX', 'TRVAL', 'TRDELT', 'COMMENT', 'OFFSET', 'SMOTOR', 'TUNIT', 'PMOTOR',
                      'LMOTOR', 'TDESC', 'NAXIS', 'TTYPE', 'TFORM', 'XTENSION', 'BITPIX', 'TDELT',
                      'TRPIX', }
        skip_predicates = {lambda k: any(s in k for s in skip_frags), }

        scan_desc = {k: v for k, v in scan_desc.items()
                    if not any(pred(k) for pred in skip_predicates)}

        data_vars = {k: (dimensions[k], hdu.data[k].reshape(spectrum_shape[k]), scan_desc)
                     for k in spectra_names}

        data_vars = rename_keys(data_vars, self.COLUMN_RENAMINGS)
        if 'beam_current' in data_vars and np.all(data_vars['beam_current'][1] == 0):
            # Wasn't taken at a beamline
            del data_vars['beam_current']

        hdulist.close()

        relevant_dimensions = {k for k in coords.keys() if k in
                               set(np.itertools.chain(*[l[0] for l in data_vars.values()]))}
        relevant_coords = {k: v for k, v in coords.items() if k in relevant_dimensions}

        dataset = xr.Dataset(
            data_vars,
            relevant_coords,
            scan_desc,
        )

        for var_name, data_arr in dataset.data_vars.items():
            if 'time' in data_arr.dims:
                data_arr.data = data_arr.sel(time=slice(None, None, -1)).data

        provenance_from_file(dataset, data_loc, {
            'what': 'Loaded Spin-ToF dataset',
            'by': 'load_DLD',
        })

        return dataset

    def load(self, scan_desc: dict=None, **kwargs):
        if scan_desc is None:
            warnings.warn('Attempting to make due without user associated scan_desc for the file')
            raise TypeError('Expected a dictionary of scan_desc with the location of the file')

        data_loc = scan_desc.get('path', scan_desc.get('file'))
        scan_desc = {k: v for k, v in scan_desc.items() if not isinstance(v, float) or not np.isnan(v)}

        if os.path.splitext(data_loc)[1] == '.fits':
            return self.load_SToF_fits(scan_desc)

        return self.load_SToF_hdf5(scan_desc)
    