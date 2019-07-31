"""
Plugin facility to read+normalize information from different sources to a common format
"""
import warnings

import numpy as np
import h5py
import xarray as xr
from astropy.io import fits

from pathlib import Path
import typing
import copy
import arpes.config
import arpes.constants
import os.path

from arpes.load_pxt import read_single_pxt, find_ses_files_associated
from arpes.utilities import rename_keys, case_insensitive_get, rename_dataarray_attrs
from arpes.preparation import replace_coords
from arpes.provenance import provenance_from_file
from arpes.endstations.fits_utils import find_clean_coords
from arpes.endstations.igor_utils import shim_wave_note
from arpes.repair import negate_energy

__all__ = ('endstation_name_from_alias', 'endstation_from_alias', 'add_endstation', 'load_scan',
           'EndstationBase', 'FITSEndstation', 'HemisphericalEndstation', 'SynchrotronEndstation',
           'SingleFileEndstation', 'load_scan_for_endstation',)

_ENDSTATION_ALIASES = {}


class EndstationBase(object):
    ALIASES = []
    PRINCIPAL_NAME = None

    # adjust as needed
    CONCAT_COORDS = ['hv', 'chi', 'psi', 'timed_power', 'tilt', 'beta', 'theta']
    SUMMABLE_NULL_DIMS = ['phi', 'cycle'] # phi because this happens sometimes at BL4 with core level scans

    RENAME_KEYS = {}

    def concatenate_frames(self, frames=typing.List[xr.Dataset], scan_desc: dict=None):
        if len(frames) == 0:
            raise ValueError('Could not read any frames.')
        elif len(frames) == 1:
            return frames[0]
        else:
            # determine which axis to stitch them together along, and then do this
            scan_coord = None
            max_different_values = -np.inf
            for possible_scan_coord in self.CONCAT_COORDS:
                coordinates = [f.attrs.get(possible_scan_coord, None) for f in frames]
                n_different_values = len(set(coordinates))
                if n_different_values > max_different_values and None not in coordinates:
                    max_different_values = n_different_values
                    scan_coord = possible_scan_coord

            assert (scan_coord is not None)

            for f in frames:
                f.coords[scan_coord] = f.attrs[scan_coord]

            frames.sort(key=lambda x: x.coords[scan_coord])
            return xr.concat(frames, scan_coord)

    def resolve_frame_locations(self, scan_desc: dict = None):
        return []

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        print(frame_path)
        return xr.Dataset()

    def postprocess(self, frame: xr.Dataset):
        frame = xr.Dataset({
            k: rename_dataarray_attrs(v, self.RENAME_KEYS) for k, v in frame.data_vars.items()
        }, attrs=rename_keys(frame.attrs, self.RENAME_KEYS))

        sum_dims = []
        for dim in frame.dims:
            if len(frame.coords[dim]) == 1 and dim in self.SUMMABLE_NULL_DIMS:
                sum_dims.append(dim)

        if len(sum_dims):
            frame = frame.sum(sum_dims, keep_attrs=True)

        return frame

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict=None):
        # attach the 'spectrum_type'
        # TODO move this logic into xarray extensions and customize here
        # only as necessary
        coord_names = tuple(sorted([c for c in data.dims if c != 'cycle']))

        spectrum_type = None
        if any(d in coord_names for d in {'x', 'y', 'z'}):
            coord_names = tuple(c for c in coord_names if c not in {'x', 'y', 'z'})
            spectrum_types = {
                ('eV',): 'spem',
                ('eV', 'phi',): 'ucut',
            }
            spectrum_type = spectrum_types.get(coord_names)
        else:
            spectrum_types = {
                ('eV',): 'xps',
                ('eV', 'phi', 'theta',): 'map',
                ('eV', 'phi', 'psi',): 'map',
                ('beta', 'eV', 'phi',): 'map',
                ('eV', 'hv', 'phi',): 'hv_map',
                ('eV', 'phi'): 'cut',
            }
            spectrum_type = spectrum_types.get(coord_names)

        if 'phi' not in data.coords:
            # XPS
            data.coords['phi'] = 0
            for s in data.S.spectra:
                s.coords['phi'] = 0

        if spectrum_type is not None:
            data.attrs['spectrum_type'] = spectrum_type
            if 'spectrum' in data.data_vars:
                data.spectrum.attrs['spectrum_type'] = spectrum_type

        ls = [data] + data.S.spectra
        for l in ls:
            for c in ['x', 'y', 'z', 'theta', 'beta', 'chi', 'hv', 'alpha', 'psi']:
                if c not in l.coords:
                    l.coords[c] = l.attrs[c]

        for l in ls:
            if 'chi' in l.coords and 'chi_offset' not in l.attrs:
                l.attrs['chi_offset'] = l.coords['chi'].item()

        return data

    def load(self, scan_desc: dict = None, **kwargs):
        """
        Loads a scan from a single file or a sequence of files.

        :param scan_desc:
        :param kwargs:
        :return:
        """
        resolved_frame_locations = self.resolve_frame_locations(scan_desc)
        resolved_frame_locations = [f if isinstance(f, str) else str(f) for f in resolved_frame_locations]

        frames = [self.load_single_frame(fpath, scan_desc, **kwargs) for fpath in resolved_frame_locations]
        frames = [self.postprocess(f) for f in frames]
        concatted = self.concatenate_frames(frames, scan_desc)
        concatted = self.postprocess_final(concatted, scan_desc)

        if 'id' in scan_desc:
            concatted.attrs['id'] = scan_desc['id']

        return concatted


class SingleFileEndstation(EndstationBase):
    def resolve_frame_locations(self, scan_desc: dict=None):
        if scan_desc is None:
            raise ValueError('Must pass dictionary as file scan_desc to all endstation loading code.')

        original_data_loc = scan_desc.get('path', scan_desc.get('file'))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        p = Path(original_data_loc)
        return [p]


class SESEndstation(EndstationBase):
    def resolve_frame_locations(self, scan_desc: dict=None):
        if scan_desc is None:
            raise ValueError('Must pass dictionary as file scan_desc to all endstation loading code.')

        original_data_loc = scan_desc.get('path', scan_desc.get('file'))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        p = Path(original_data_loc)
        return find_ses_files_associated(p)

    def load_single_frame(self, frame_path: str=None, scan_desc: dict=None, **kwargs):
        name, ext = os.path.splitext(frame_path)

        if 'nc' in ext:
            # was converted to hdf5/NetCDF format with Conrad's Igor scripts
            scan_desc = copy.deepcopy(scan_desc)
            scan_desc['path'] = frame_path
            return self.load_SES_nc(scan_desc=scan_desc, **kwargs)

        # it's given by SES PXT files
        pxt_data = negate_energy(read_single_pxt(frame_path))
        return xr.Dataset({'spectrum': pxt_data}, attrs=pxt_data.attrs)

    def postprocess(self, frame: xr.Dataset):
        import arpes.xarray_extensions

        frame = super().postprocess(frame)
        return frame.assign_attrs(frame.S.spectrum.attrs)

    def load_SES_nc(self, scan_desc: dict=None, robust_dimension_labels=False, **kwargs):
        """
        Imports an hdf5 dataset exported from Igor that was originally generated by a Scienta spectrometer
        in the SESb format. In order to understand the structure of these files have a look at Conrad's
        saveSESDataset in Igor Pro.

        :param scan_desc: Dictionary with extra information to attach to the xr.Dataset, must contain the location
        of the file
        :return: xr.Dataset
        """

        scan_desc = copy.deepcopy(scan_desc)

        data_loc = scan_desc.get('path', scan_desc.get('file'))
        p = Path(data_loc)
        if not p.exists():
            data_loc = os.path.join(arpes.config.DATA_PATH, data_loc)

        wave_note = shim_wave_note(data_loc)
        f = h5py.File(data_loc, 'r')

        primary_dataset_name = list(f)[0]
        # This is bugged for the moment in h5py due to an inability to read fixed length unicode strings
        # wave_note = f['/' + primary_dataset_name].attrs['IGORWaveNote']

        # Use dimension labels instead of
        dimension_labels = list(f['/' + primary_dataset_name].attrs['IGORWaveDimensionLabels'][0])
        if any(x == '' for x in dimension_labels):
            print(dimension_labels)

            if not robust_dimension_labels:
                raise ValueError('Missing dimension labels. Use robust_dimension_labels=True to override')
            else:
                used_blanks = 0
                for i in range(len(dimension_labels)):
                    if dimension_labels[i] == '':
                        dimension_labels[i] = 'missing{}'.format(used_blanks)
                        used_blanks += 1

                print(dimension_labels)

        scaling = f['/' + primary_dataset_name].attrs['IGORWaveScaling'][-len(dimension_labels):]
        raw_data = f['/' + primary_dataset_name][:]

        scaling = [np.linspace(scale[1], scale[1] + scale[0] * raw_data.shape[i], raw_data.shape[i])
                   for i, scale in enumerate(scaling)]

        dataset_contents = {}
        attrs = scan_desc.pop('note', {})
        attrs.update(wave_note)

        built_coords = dict(zip(dimension_labels, scaling))

        deg_to_rad_coords = {'theta', 'beta', 'phi', 'alpha', 'psi'}

        # the hemisphere axis is handled below
        built_coords = {k: c * (np.pi / 180) if k in deg_to_rad_coords else c
                        for k, c in built_coords.items()}

        deg_to_rad_attrs = {'theta', 'beta', 'alpha', 'psi', 'chi'}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in attrs:
                attrs[angle_attr] = float(attrs[angle_attr]) * np.pi / 180

        dataset_contents['spectrum'] = xr.DataArray(
            raw_data,
            coords=built_coords,
            dims=dimension_labels,
            attrs=attrs,
        )

        provenance_from_file(dataset_contents['spectrum'], data_loc, {
            'what': 'Loaded SES dataset from HDF5.',
            'by': 'load_SES'
        })

        return xr.Dataset(
            dataset_contents,
            attrs={**scan_desc, 'name': primary_dataset_name},
        )


class FITSEndstation(EndstationBase):
    PREPPED_COLUMN_NAMES = {
        'time': 'time',
        'Delay': 'delay-var',  # these are named thus to avoid conflicts with the
        'Sample-X': 'cycle-var',  # underlying coordinates
        'Mira': 'pump_power',
        # insert more as needed
    }

    SKIP_COLUMN_NAMES = {
        'Phi',
        'null',
        'X',
        'Y',
        'Z',
        'mono_eV',
        'Slit Defl',
        'Optics Stage',
        'Scan X',
        'Scan Y',
        'Scan Z',
        # insert more as needed
    }

    SKIP_COLUMN_FORMULAS = {
        lambda name: True if ('beamview' in name or 'IMAQdx' in name) else False,
    }

    RENAME_KEYS = {
        'Phi': 'chi',
        'Beta': 'beta',
        'Azimuth': 'chi',
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
        'LMOTOR6': 'alpha',
    }

    def resolve_frame_locations(self, scan_desc: dict=None):
        if scan_desc is None:
            raise ValueError('Must pass dictionary as file scan_desc to all endstation loading code.')

        original_data_loc = scan_desc.get('path', scan_desc.get('file'))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        return [original_data_loc]

    def load_single_frame(self, frame_path: str=None, scan_desc: dict=None, **kwargs):
        # Use dimension labels instead of
        hdulist = fits.open(frame_path, ignore_missing_end=True)
        primary_dataset_name = None

        # Clean the header because sometimes out LabView produces improper FITS files
        for i in range(len(hdulist)):
            # This looks a little stupid, but because of confusing astropy internals actually works
            hdulist[i].header['UN_0_0'] = ''  # TODO This card is broken, this is not a good fix
            del hdulist[i].header['UN_0_0']
            hdulist[i].header['UN_0_0'] = ''
            if 'TTYPE2' in hdulist[i].header and hdulist[i].header['TTYPE2'] == 'Delay':
                hdulist[i].header['TUNIT2'] = ''
                del hdulist[i].header['TUNIT2']
                hdulist[i].header['TUNIT2'] = 'ps'

            hdulist[i].verify('fix+warn')
            hdulist[i].header.update()
            # This actually requires substantially more work because it is lossy to information
            # on the unit that was encoded

        hdu = hdulist[1]

        scan_desc = copy.deepcopy(scan_desc)
        attrs = scan_desc.pop('note', scan_desc)
        attrs.update(dict(hdulist[0].header))

        drop_attrs = ['COMMENT', 'HISTORY', 'EXTEND', 'SIMPLE', 'SCANPAR', 'SFKE_0']
        for dropped_attr in drop_attrs:
            if dropped_attr in attrs:
                del attrs[dropped_attr]

        built_coords, dimensions, real_spectrum_shape = find_clean_coords(hdu, attrs, mode='MC')
        attrs = rename_keys(attrs, self.RENAME_KEYS)
        scan_desc = rename_keys(scan_desc, self.RENAME_KEYS)

        def clean_key_name(k):
            if '#' in k:
                k = k.replace('#', 'num')

            return k

        attrs = {clean_key_name(k): v for k, v in attrs.items()}
        scan_desc = {clean_key_name(k): v for k, v in scan_desc.items()}

        # don't have phi because we need to convert pixels first
        deg_to_rad_coords = {'beta', 'theta', 'chi'}

        # convert angular attributes to radians
        for coord_name in deg_to_rad_coords:
            if coord_name in attrs:
                try:
                    attrs[coord_name] = float(attrs[coord_name]) * (np.pi / 180)
                except (TypeError, ValueError):
                    pass
            if coord_name in scan_desc:
                try:
                    scan_desc[coord_name] = float(scan_desc[coord_name]) * (np.pi / 180)
                except (TypeError, ValueError):
                    pass

        data_vars = {}

        all_names = hdu.columns.names
        n_spectra = len([n for n in all_names if 'Fixed_Spectra' in n or 'Swept_Spectra' in n])
        for column_name in hdu.columns.names:
            # we skip some fixed set of the columns, such as the one dimensional axes, as well as things that are too
            # tricky to load at the moment, like the microscope images from MAESTRO
            should_skip = False
            if column_name in self.SKIP_COLUMN_NAMES:
                should_skip = True

            for formula in self.SKIP_COLUMN_FORMULAS:
                if formula(column_name):
                    should_skip = True

            if should_skip:
                continue

            # the hemisphere axis is handled below
            dimension_for_column = dimensions[column_name]
            column_shape = real_spectrum_shape[column_name]

            column_display = self.PREPPED_COLUMN_NAMES.get(column_name, column_name)
            if 'Fixed_Spectra' in column_display:
                if n_spectra == 1:
                    column_display = 'spectrum'
                else:
                    column_display = 'spectrum' + '-' + column_display.split('Fixed_Spectra')[1]

            if 'Swept_Spectra' in column_display:
                if n_spectra == 1:
                    column_display = 'spectrum'
                else:
                    column_display = 'spectrum' + '-' + column_display.split('Swept_Spectra')[1]

            # sometimes if a scan is terminated early it can happen that the sizes do not match the expected value
            # as an example, if a beta map is supposed to have 401 slices, it might end up having only 260 if it were
            # terminated early
            # If we are confident in our parsing code above, we can handle this case and take a subset of the coords
            # so that the data matches
            try:
                resized_data = hdu.data.columns[column_name].array.reshape(column_shape)
            except ValueError:
                # if we could not resize appropriately, we will try to reify the shapes together
                rest_column_shape = column_shape[1:]
                n_per_slice = int(np.prod(rest_column_shape))
                total_shape = hdu.data.columns[column_name].array.shape
                total_n = np.prod(total_shape)

                n_slices = total_n // n_per_slice
                # if this isn't true, we can't recover
                data_for_resize = hdu.data.columns[column_name].array
                if (total_n // n_per_slice != total_n / n_per_slice):
                    # the last slice was in the middle of writing when something hit the fan
                    # we need to infer how much of the data to read, and then repeat the above
                    # we need to cut the data

                    # This can happen when the labview crashes during data collection,
                    # we use column_shape[1] because of the row order that is used in the FITS file
                    data_for_resize = data_for_resize[0:(total_n // n_per_slice) * column_shape[1]]
                    warnings.warn(
                        'Column {} was in the middle of slice when DAQ stopped. Throwing out incomplete slice...'.format(
                            column_name))

                column_shape = list(column_shape)
                column_shape[0] = n_slices

                try:
                    resized_data = data_for_resize.reshape(column_shape)
                except Exception:
                    # sometimes for whatever reason FITS errors and cannot read the data
                    continue

                # we also need to adjust the coordinates
                altered_dimension = dimension_for_column[0]
                built_coords[altered_dimension] = built_coords[altered_dimension][:n_slices]

            data_vars[column_display] = xr.DataArray(
                resized_data,
                coords={k: c for k, c in built_coords.items() if k in dimension_for_column},
                dims=dimension_for_column,
                attrs=attrs,
            )

        def prep_spectrum(data: xr.DataArray):
            # don't do center pixel inference because the main chamber
            # at least consistently records the offset from the edge
            # of the recorded window
            if 'pixel' in data.coords:
                phi_axis = data.coords['pixel'].values * \
                           arpes.constants.SPECTROMETER_MC['rad_per_pixel']
                data = replace_coords(data, {
                    'phi': phi_axis
                }, [('pixel', 'phi',)])

            # Always attach provenance
            provenance_from_file(data, frame_path, {
                'what': 'Loaded MC dataset from FITS.',
                'by': 'load_MC',
            })

            return data

        if 'spectrum' in data_vars:
            data_vars['spectrum'] = prep_spectrum(data_vars['spectrum'])

        # adjust angular coordinates
        built_coords = {k: c * (np.pi / 180) if k in deg_to_rad_coords else c
                        for k, c in built_coords.items()}

        return xr.Dataset(
            data_vars,
            attrs={**scan_desc, 'name': primary_dataset_name},
        )


class SynchrotronEndstation(EndstationBase):
    RESOLUTION_TABLE = None

class HemisphericalEndstation(EndstationBase):
    """
    An endstation definition for a hemispherical analyzer should include
    everything needed to determine energy + k resolution, angle conversion,
    and ideally correction databases for dead pixels + detector nonlinearity
    information
    """
    ANALYZER_INFORMATION = None
    SLIT_ORIENTATION = None
    PIXELS_PER_DEG = None


def endstation_name_from_alias(alias):
    return _ENDSTATION_ALIASES[alias].PRINCIPAL_NAME


def endstation_from_alias(alias):
    return _ENDSTATION_ALIASES[alias]


def add_endstation(endstation_cls):
    # add the aliases
    assert(endstation_cls.PRINCIPAL_NAME is not None)
    for alias in endstation_cls.ALIASES:
        if alias in _ENDSTATION_ALIASES:
            continue
            print('Alias ({}) already registered. Skipping...'.format(alias))

        _ENDSTATION_ALIASES[alias] = endstation_cls

    if endstation_cls.PRINCIPAL_NAME in _ENDSTATION_ALIASES and endstation_cls.PRINCIPAL_NAME not in endstation_cls.ALIASES:
        # indicates it was added earlier, so there's an alias conflict
        if False:
            warnings.warn('Endstation name or alias conflicts with existing {}'.format(endstation_cls.PRINCIPAL_NAME))

    _ENDSTATION_ALIASES[endstation_cls.PRINCIPAL_NAME] = endstation_cls


def load_scan_for_endstation(scan_desc, endstation_cls, **kwargs):
    note = scan_desc.get('note', scan_desc)
    full_note = copy.deepcopy(scan_desc)
    full_note.update(note)

    return endstation_cls().load(scan_desc, **kwargs)

def load_scan(scan_desc, **kwargs):
    note = scan_desc.get('note', scan_desc)
    full_note = copy.deepcopy(scan_desc)
    full_note.update(note)

    endstation_name = case_insensitive_get(full_note, 'location', case_insensitive_get(full_note, 'endstation'))
    try:
        endstation_cls = endstation_from_alias(endstation_name)
    except KeyError:
        raise ValueError('Could not identify endstation. '
                         'Did you set the endstation or location? Find a description of the available options '
                         'in the endstations module.')

    return endstation_cls().load(scan_desc, **kwargs)