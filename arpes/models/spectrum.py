# TODO Reorganize code in here. You should be able to register new loading code in order to
# make this code more portable to other beamlines and spectrometers.

import copy
import itertools
import math
import os.path
import subprocess
import warnings
from ast import literal_eval
from collections import Iterable

import h5py
import numpy
import xarray
from astropy.io import fits
from scipy import ndimage

import arpes.config
import arpes.constants as consts
from arpes.exceptions import AnalysisError
from arpes.preparation import infer_center_pixel, replace_coords
from arpes.preparation.tof_preparation import convert_SToF_to_energy
from arpes.provenance import provenance_from_file
from arpes.utilities import split_hdu_header, rename_keys
from .viewable import Viewable

__all__ = ['load_scan']


_RENAME_DIMS = {
    'Beta': 'polar',
    'Theta': 'phi',
    'Delay': 'delay',
    'Sample-X': 'cycle',
    'null': 'cycle',
}

def shim_wave_note(path):
    """
    Hack to read the corrupted wavenote out of the h5 files that Igor has been producing.
    h5 dump still produces the right value, so we use it from the command line in order to get the value of the note.
    :param path: Location of the file
    :return:
    """
    wave_name = os.path.splitext(os.path.basename(path))[0]
    cmd = 'h5dump -A --attribute /{}/IGORWaveNote {}'.format(wave_name, path)
    h5_out = subprocess.getoutput(cmd)

    split_data = h5_out[h5_out.index('DATA {'):]
    assert(len(split_data.split('"')) == 3)
    data = split_data.split('"')[1]

    # remove stuff below the end of the header
    try:
        data = data[:data.index('ENDHEADER')]
    except ValueError:
        pass

    lines = [l.strip() for l in data.splitlines() if '=' in l]
    lines = itertools.chain(*[l.split(',') for l in lines])
    return dict([l.split('=') for l in lines])


def load_scan(scan_desc):
    # TODO support other spectrometers and scan types transparently
    if 'SES' in scan_desc.get('note', scan_desc).get('Instrument', ''):
        return load_SES(scan_desc)

    location_candidate = scan_desc.get('note', scan_desc).get('location', '')
    load_fn = {
        'BL403': load_SES,
        'ALG-MC': load_MC,
        'ALG-SToF': load_SToF,
    }.get(location_candidate)

    if load_fn is not None:
        return load_fn(scan_desc)

    raise ValueError('Could not identify appropriate spectrometer')


def load_MC(metadata: dict=None, filename: str=None):
    """
    Import a FITS dataset produced by the main chamber LabView.

    The metadata for scans conducted in this way typically is stqqored in an
    Excel file.
    :param metadata: Dictionary with extra information to attach to the xarray.Dataset, must contain the location
    of the file
    :return:
    """

    if metadata is None and filename is None:
        warnings.warn('Attempting to make due without user associated metadata for the file')
        raise TypeError('Expected a dictionary of metadata with the location of the file')

    if metadata is None:
        metadata = {'file': filename}

    metadata = copy.deepcopy(metadata)

    data_loc = metadata.get('path', metadata.get('file'))
    data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)


    # Use dimension labels instead of
    hdulist = fits.open(data_loc, ignore_missing_end=True)
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

    dataset_contents = dict()
    attrs = metadata.pop('note', metadata)
    attrs.update(dict(hdulist[0].header))

    drop_attrs = ['COMMENT', 'HISTORY', 'EXTEND', 'SIMPLE', 'SCANPAR', 'SFKE_0']
    for dropped_attr in drop_attrs:
        if dropped_attr in attrs:
            del attrs[dropped_attr]

    if 'id' in metadata:
        attrs['id'] = metadata['id']

    built_coords, dimensions, real_spectrum_shape = find_clean_coords(hdu, attrs)
    warnings.warn('Not loading all available spectra for main chamber scan. TODO.')

    dimensions = dimensions[hdu.columns.names[-1]]
    real_spectrum_shape = real_spectrum_shape[hdu.columns.names[-1]]

    dataset_contents['raw'] = xarray.DataArray(
        hdu.data.columns['Fixed_Spectra4'].array.reshape(real_spectrum_shape),
        coords=built_coords,
        dims=dimensions,
        attrs=attrs,
    )

    center_pixel = infer_center_pixel(dataset_contents['raw'])
    phi_axis = (dataset_contents['raw'].coords['pixel'].values - center_pixel) * \
               arpes.constants.SPECTROMETER_MC['deg_per_pixel']
    dataset_contents['raw'] = replace_coords(dataset_contents['raw'], {
        'phi': phi_axis
    }, [('pixel', 'phi',)])

    provenance_from_file(dataset_contents['raw'], data_loc, {
        'what': 'Loaded MC dataset from FITS.',
        'by': 'load_MC',
    })

    return xarray.Dataset(
        dataset_contents,
        attrs={**metadata, 'name': primary_dataset_name},
    )


def find_clean_coords(hdu, attrs, spectra=None):
    """
    Determines the scan degrees of freedom, the shape of the actual "spectrum"
    and reads and parses the coordinates from the header information in the recorded
    scan.

    TODO Write data loading tests to ensure we don't break MC compatibility

    :param hdu:
    :param attrs:
    :return: (coordinates, dimensions, np shape of actual spectrum)
    """
    scan_coords, scan_dimension, scan_shape = extract_coords(attrs)
    scan_dimension = [_RENAME_DIMS.get(s, s) for s in scan_dimension[::-1]]
    scan_coords = {_RENAME_DIMS.get(k, k): v for k, v in scan_coords.items()}
    extra_coords = {}
    scan_shape = scan_shape[::-1]

    spectrum_shapes = {}
    dimensions_for_spectra = {}

    if spectra is None:
        spectra = hdu.columns.names

    if isinstance(spectra, str):
        spectra = [spectra]

    for spectrum_key in spectra:
        if spectrum_key is None:
            spectrum_key = hdu.columns.names[-1]

        if isinstance(spectrum_key, str):
            spectrum_key = hdu.columns.names.index(spectrum_key) + 1

        spectrum_name = hdu.columns.names[spectrum_key - 1]

        try:
            offset = hdu.header['TRVAL%g' % spectrum_key]
            delta = hdu.header['TDELT%g' % spectrum_key]
            offset = literal_eval(offset) if isinstance(offset, str) else offset
            delta = literal_eval(delta) if isinstance(delta, str) else delta

            try:
                shape = hdu.header['TDIM%g' % spectrum_key]
                shape = literal_eval(shape) if isinstance(shape, str) else shape
            except:
                shape = hdu.data.field(spectrum_key - 1).shape

            if not isinstance(delta, Iterable):
                delta = (delta,)
            if not isinstance(offset, Iterable):
                offset = (offset,)

        except KeyError:
            # if TRVAL{spectrum_key} was not found this means that this column is scalar,
            # i.e. it has only one value at any point in the scan
            spectrum_shapes[spectrum_name] = scan_shape
            dimensions_for_spectra[spectrum_name] = scan_dimension
            continue


        if len(scan_shape) == 0 and shape[0] == 1:
            # the ToF pads with ones on single EDCs
            shape = shape[1:]

        rest_shape = shape[len(scan_shape):]

        assert(len(offset) == len(delta) and len(delta) == len(rest_shape))

        coords = zip(offset, delta, shape)
        coords = [numpy.linspace(o, o + s * d, s, endpoint=False)
                  for o, d, s in zip(offset, delta, rest_shape)]

        coord_names_for_spectrum = {
            'Time_Spectra': ['time'],
            'Energy_Spectra': ['eV'],
            'Fixed_Spectra4': ['pixel', 'phi'], # MC hemisphere image
            'wave':  ['time'],
            'targetPlus': ['time'],
            'targetMinus': ['time'],
        }

        if spectrum_name not in coord_names_for_spectrum:
            # Don't remember what the MC ones were, so I will wait to do those again
            # Might have to add new items for new spectrometers as well
            import pdb
            pdb.set_trace()

        coords_for_spectrum = dict(zip(coord_names_for_spectrum[spectrum_name], coords))
        extra_coords.update(coords_for_spectrum)
        dimensions_for_spectra[spectrum_name] = \
            tuple(scan_dimension) + tuple(coord_names_for_spectrum[spectrum_name])
        spectrum_shapes[spectrum_name] = tuple(scan_shape) + tuple(rest_shape)
        coords_for_spectrum.update(scan_coords)

    extra_coords.update(scan_coords)
    return extra_coords, dimensions_for_spectra, spectrum_shapes


def extract_coords(attrs):
    """
    Does the hard work of extracting coordinates from the scan description.
    :param attrs:
    :return:
    """
    try:
        n_loops = attrs['LWLVLPN']
    except KeyError:
        # Looks like no scan, this happens for instance in the SToF when you take a single
        # EDC
        return {}, [], (),
    scan_dimension = []
    scan_shape = []
    scan_coords = {}
    for loop in range(n_loops):
        n_scan_dimensions = attrs['NMSBDV%g' % loop]
        if attrs['SCNTYP%g' % loop] == 0:  # computed
            for i in range(n_scan_dimensions):
                name, start, end, n = (
                    attrs['NM_%g_%g' % (loop, i,)],
                    # attrs['UN_0_%g' % i],
                    float(attrs['ST_%g_%g' % (loop, i,)]),
                    float(attrs['EN_%g_%g' % (loop, i,)]),
                    int(attrs['N_%g_%g' % (loop, i,)]),
                )

                name = _RENAME_DIMS.get(name, name)

                scan_dimension.append(name)
                scan_shape.append(n)
                scan_coords[name] = numpy.linspace(start, end, n, endpoint=True)
        else:  # tabulated scan, this is more complicated
            name, n = (
                attrs['NM_%g_0' % loop],
                attrs['NMPOS_%g' % loop],
            )
            n_regions_key = {'Delay': 'DS_NR'}.get(name, 'DS_NR')
            n_regions = attrs[n_regions_key]

            name = _RENAME_DIMS.get(name, name)

            coord = numpy.array(())
            for region in range(n_regions):
                start, end, n = (
                    attrs['ST_%g_%g' % (loop, region,)],
                    attrs['EN_%g_%g' % (loop, region,)],
                    attrs['N_%g_%g' % (loop, region,)],
                )

                coord = numpy.concatenate((coord, numpy.linspace(
                    start, end, n, endpoint=True),))

            scan_dimension.append(name)
            scan_shape.append(len(coord))
            scan_coords[name] = coord
    return scan_coords, scan_dimension, scan_shape


def load_SES(metadata: dict=None, filename: str=None):
    """
    Imports an hdf5 dataset exported from Igor that was originally generated by a Scienta spectrometer
    in the SESb format. In order to understand the structure of these files have a look at Conrad's
    saveSESDataset in Igor Pro.

    :param metadata: Dictionary with extra information to attach to the xarray.Dataset, must contain the location
    of the file
    :return: xarray.Dataset
    """

    if metadata is None and filename is None:
        warnings.warn('Attempting to make due without user associated metadata for the file')
        raise TypeError('Expected a dictionary of metadata with the location of the file')

    if metadata is None:
        metadata = {'file': filename}

    metadata = copy.deepcopy(metadata)

    data_loc = metadata.get('path', metadata.get('file'))
    data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

    wave_note = shim_wave_note(data_loc)
    f = h5py.File(data_loc, 'r')

    primary_dataset_name = list(f)[0]
    # This is bugged for the moment in h5py due to an inability to read fixed length unicode strings
    # wave_note = f['/' + primary_dataset_name].attrs['IGORWaveNote']

    # Use dimension labels instead of
    dimension_labels = f['/' + primary_dataset_name].attrs['IGORWaveDimensionLabels'][0]
    scaling = f['/' + primary_dataset_name].attrs['IGORWaveScaling'][-len(dimension_labels):]

    raw_data = f['/' + primary_dataset_name][:]

    scaling = [numpy.linspace(scale[1], scale[1] + scale[0] * raw_data.shape[i], raw_data.shape[i])
               for i, scale in enumerate(scaling)]

    dataset_contents = dict()
    attrs = metadata.pop('note', {})
    attrs.update(wave_note)
    if 'id' in metadata:
        attrs['id'] = metadata['id']

    dataset_contents['raw'] = xarray.DataArray(
        raw_data,
        coords=dict(zip(dimension_labels, scaling)),
        dims=dimension_labels,
        attrs=attrs,
    )

    provenance_from_file(dataset_contents['raw'], data_loc, {
        'what': 'Loaded SES dataset from HDF5.',
        'by': 'load_SES'
    })

    return xarray.Dataset(
        dataset_contents,
        attrs={**metadata, 'name': primary_dataset_name},
    )


def load_DLD(metadata: dict=None, filename: str=None):
    """
    Imports a FITS file that contains all of the information from a run of Ping
    and Anton's delay line detector ARToF

    :param metadata: Dictionary with extra information to attach to the xarray.Dataset, must contain the location
    of the file
    :return: xarray.Dataset
    """

    if metadata is None and filename is None:
        warnings.warn('Attempting to make due without user associated metadata for the file')
        raise TypeError('Expected a dictionary of metadata with the location of the file')

    if metadata is None:
        metadata = {'file': filename}

    metadata = copy.deepcopy(metadata)

    data_loc = metadata['file']
    data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

    f = h5py.File(data_loc, 'r')

    dataset_contents = dict()
    raw_data = f['/PRIMARY/DATA'][:]
    raw_data = raw_data[:,::-1] # Reverse the timing axis
    dataset_contents['raw'] = xarray.DataArray(
        raw_data,
        coords={'x_pixels': numpy.linspace(0, 511, 512),
                't_pixels': numpy.linspace(0, 511, 512)},
        dims=('x_pixels', 't_pixels'),
        attrs=f['/PRIMARY'].attrs.items(),
    )

    provenance_from_file(dataset_contents['raw'], data_loc, {
        'what': 'Loaded Anton and Ping DLD dataset from HDF5.',
        'by': 'load_DLD',
    })

    return xarray.Dataset(
        dataset_contents,
        attrs=metadata
    )


def load_SToF(metadata: dict=None, filename: str=None):
    if metadata is None and filename is None:
        warnings.warn('Attempting to make due without user associated metadata for the file')
        raise TypeError('Expected a dictionary of metadata with the location of the file')

    if metadata is None:
        metadata = {'file': filename}

    data_loc = metadata.get('path', metadata.get('file'))
    metadata = {k: v for k, v in metadata.items() if not isinstance(v, float) or not numpy.isnan(v)}

    if os.path.splitext(data_loc)[1] == '.fits':
        return load_SToF_fits(metadata, filename)

    return load_SToF_hdf5(metadata, filename)


def load_SToF_fits(metadata: dict=None, filename: str=None):
    if metadata is None and filename is None:
        warnings.warn('Attempting to make due without user associated metadata for the file')
        raise TypeError('Expected a dictionary of metadata with the location of the file')

    if metadata is None:
        metadata = {'file': filename}

    metadata = dict(copy.deepcopy(metadata))

    data_loc = metadata.get('path', metadata.get('file'))
    data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

    hdulist = fits.open(data_loc)

    hdulist[0].verify('fix+warn')
    header_hdu, hdu = hdulist[0], hdulist[1]

    metadata.update(dict(hdu.header))
    metadata.update(dict(header_hdu.header))

    drop_attrs = ['COMMENT', 'HISTORY', 'EXTEND', 'SIMPLE', 'SCANPAR', 'SFKE_0']
    for dropped_attr in drop_attrs:
        if dropped_attr in metadata:
            del metadata[dropped_attr]

    coords, dimensions, spectrum_shape = find_clean_coords(hdu, metadata)

    columns = hdu.columns
    dataset = None

    column_renamings = {
        'TempA': 'temperature_cryo',
        'TempB': 'temperature_sample',
        'Current': 'photocurrent',
        'ALS_Beam_mA': 'beam_current',
        'Energy_Spectra': 'spectrum',
        'targetPlus': 'up',
        'targetMinus': 'down',
        'wave': 'spectrum', # this should not occur simultaneously with 'Energy_Spectra'
    }

    is_spin_resolved = 'targetPlus' in columns.names or 'targetMinus' in columns.names
    requires_conversion = is_spin_resolved or 'wave' in columns.names
    spin_columns = ['targetPlus', 'targetMinus', 'Current' 'TempA', 'TempB', 'ALS_Beam_mA']
    straight_columns = ['Current', 'TempA', 'TempB', 'ALS_Beam_mA', 'Energy_Spectra', 'wave']
    take_columns = spin_columns if is_spin_resolved else straight_columns

    # We could do our own spectrum conversion too, but that would be more annoying
    # it would slightly improve accuracy though
    spectra_names = [name for name in take_columns if name in columns.names]
    #}, column_renamings)

    skip_frags = {'MMX', 'TRVAL', 'TRDELT', 'COMMENT', 'OFFSET', 'SMOTOR', 'TUNIT', 'PMOTOR',
                  'LMOTOR', 'TDESC', 'NAXIS', 'TTYPE', 'TFORM', 'XTENSION', 'BITPIX', 'TDELT',
                  'TRPIX', }
    skip_predicates = {lambda k: any(s in k for s in skip_frags), }

    metadata = {k: v for k, v in metadata.items()
                if not any(pred(k) for pred in skip_predicates)}

    data_vars = {k: (dimensions[k], hdu.data[k].reshape(spectrum_shape[k]), metadata)
                 for k in spectra_names}

    data_vars = rename_keys(data_vars, column_renamings)
    if 'beam_current' in data_vars and numpy.all(data_vars['beam_current'][1] == 0):
        # Wasn't taken at a beamline
        del data_vars['beam_current']

    hdulist.close()

    relevant_dimensions = {k for k in coords.keys() if k in
                           set(itertools.chain(*[l[0] for l in data_vars.values()]))}
    relevant_coords = {k: v for k, v in coords.items() if k in relevant_dimensions}


    dataset = xarray.Dataset(
        data_vars,
        relevant_coords,
        metadata,
    )

    for var_name, data_arr in dataset.data_vars.items():
        if 'time' in data_arr.dims:
            data_arr.data = data_arr.sel(time=slice(None, None, -1)).data

    if requires_conversion:
        dataset = convert_SToF_to_energy(dataset)

    provenance_from_file(dataset, data_loc, {
        'what': 'Loaded Spin-ToF dataset',
        'by': 'load_DLD',
    })

    return dataset


def load_SToF_hdf5(metadata: dict=None, filename: str=None):
    """
    Imports a FITS file that contains ToF spectra.

    :param metadata: Dictionary with extra information to attach to the xarray.Dataset, must contain the location
    of the file
    :return: xarray.Dataset
    """

    if metadata is None and filename is None:
        warnings.warn('Attempting to make due without user associated metadata for the file')
        raise TypeError('Expected a dictionary of metadata with the location of the file')

    if metadata is None:
        metadata = {'file': filename}

    metadata = copy.deepcopy(metadata)

    data_loc = metadata.get('path', metadata.get('file'))
    data_loc = data_loc if data_loc.startswith('/') else os.path.join(arpes.config.DATA_PATH, data_loc)

    f = h5py.File(data_loc, 'r')

    dataset_contents = dict()
    raw_data = f['/PRIMARY/DATA'][:]
    raw_data = raw_data[:,::-1] # Reverse the timing axis
    dataset_contents['raw'] = xarray.DataArray(
        raw_data,
        coords={'x_pixels': numpy.linspace(0, 511, 512),
                't_pixels': numpy.linspace(0, 511, 512)},
        dims=('x_pixels', 't_pixels'),
        attrs=f['/PRIMARY'].attrs.items(),
    )

    provenance_from_file(dataset_contents['raw'], data_loc, {
        'what': 'Loaded Anton and Ping DLD dataset from HDF5.',
        'by': 'load_DLD',
    })

    return xarray.Dataset(
        dataset_contents,
        attrs=metadata
    )


def BL10_Spectrum3D_factory(metadata=None):
    """
    TODO This method probably has more knowledge about the internals of FITs files
    than it should, we should encapsulate that knowledge more appropriately somewhere
    else in a way sensitive to the differences between chambers and beamlines.
    """
    if metadata is None:
        raise TypeError('Expected a dictionary of metadata with the location of ' +
                        'the FITs file')

    metadata = copy.deepcopy(metadata)

    hdulist = fits.open(metadata['file'])
    header = hdulist[1].header

    n_spectra = len(hdulist[1].data)
    individual_spectrum_shape_raw = hdulist[1].data[0][2].shape
    individual_spectrum_shape_reported = tuple(
        int(i) for i in split_hdu_header(header['TDIM3']))

    # sometimes we have to transpose the data for some reason...
    transpose_spectra = individual_spectrum_shape_raw != individual_spectrum_shape_reported

    full_shape = tuple(itertools.chain([n_spectra], individual_spectrum_shape_reported))
    data = numpy.ndarray(shape=full_shape, dtype=float)

    # copy the data in
    for i in range(n_spectra):
        spectrum = hdulist[1].data[i][2]
        if transpose_spectra:
            spectrum = spectrum.T

        # merge it
        data[i] = spectrum

    # now that we've collected the raw data, we need to infer bounds
    # on the data since we will soon be discarding the fits file
    name_replacements = {
        'pixels': 'pixel',
        'pxs': 'pixel',
        'eV': 'E',
    }

    axes = []
    bounds = []
    # get the first axis information from the metadata
    axes.append(metadata['scan_axis'])
    bounds.append(metadata['scan_axis']['bounds'])

    for i in range(len(full_shape) - 1):
        raw_name = split_hdu_header(header['TDESC3'])[i]
        delta = float(split_hdu_header(header['TDELT3'])[i])
        lower_bound = float(split_hdu_header(header['TRVAL3'])[i])
        n = int(split_hdu_header(header['TDIM3'])[i])
        upper_bound = lower_bound + delta * (n - 1)
        bounds_for_axis = (
            lower_bound,
            upper_bound,
            delta,
        )
        axes.append({
            'name': name_replacements.get(raw_name, raw_name),
            'bounds': bounds_for_axis,
        })
        bounds.append(bounds_for_axis)

    # we're all done, so make sure to clean up after ourselves and build the
    # spectrum, incidentally, I'm not sure fits.open provides a context manager
    # so we have to do this manually as opposed to operating inside a 'with'
    # block
    hdulist.close()
    return Spectrum3D(data=data, axes=axes, bounds=bounds, metadata=metadata)

class SToFEDC(object):
    def __init__(self, data=None, variables=None, header=None, metadata=None,
                 **kwargs): #pylint: disable=W0613
        self.length = consts.SPIN_TOF_LENGTH
        self.metadata = metadata
        self.data = data
        self.variables = variables
        self.header = header
        self.reported_t0 = metadata['t0']
        self.dE = metadata['dE']
        self.time_res = self.header['TDELT1']

        # initialize spectra
        self.spin_up_time = data['up']
        self.spin_down_time = data['down']
        self.spin_up_energy = None
        self.spin_down_energy = None
        self.raw_timing = numpy.linspace(
            0, (len(self.spin_up_time) -1) * self.time_res, len(self.spin_up_time))
        self.timing = None

        photon_offset = 1e9 * (self.length / 3e8)
        self.t0 = self.reported_t0 + photon_offset

        # All data needs to be made relative to t0
        self.adjust_timing()
        self.convert_to_kinetic_energy()

    def adjust_timing(self):
        t_diff = self.t0 - self.raw_timing[-1]
        if t_diff < 0:
            t0_index = numpy.searchsorted(self.raw_timing, self.t0)
            timing = self.raw_timing[:t0_index]
            self.spin_up_time = self.spin_up_time[:t0_index]
            self.spin_down_time = self.spin_down_time[:t0_index]
        else:
            timing = self.raw_timing

        self.timing = timing
        self.spin_up_time = self.spin_up_time[::-1]
        self.spin_down_time = self.spin_down_time[::-1]

    def convert_to_kinetic_energy(self):
        """
        Convert the ToF timing information into an energy histogram

        The core of these routines come from the Igor procedures in
        ``LoadTOF7_3.51.ipf``.

        To explain in a bit more detail what actaully happens, this
        function essentially

        1. Determines the time to energy conversion factor
        2. Calculates the requested energy binning range and allocates arrays
        3. Rebins a time spectrum into an energy spectrum, preserving the
           spectral weight, this requires a modicum of care around splitting
           counts at the edges of the new bins.
        """

        # This should be simplified
        # c = (0.5) * (9.11e-31) * self.mstar * (self.length ** 2) / (1.6e-19) * (1e18)
        # Removed factors of ten and substituted mstar = 0.5
        c = (0.5) * (9.11e6) * 0.5 * (self.length ** 2) / 1.6
        t_min, t_max = self.timing[0], self.timing[-1]
        E_min = c / (t_max ** 2)
        E_max = self.metadata['E_max']
        print(E_min, E_max, c)
        E_min_R = (math.floor(E_min/self.dE) + 1) * self.dE
        E_max_p = c / (t_min ** 2)

        if E_max_p >= E_max:
            E_max_R = math.floor(E_max/self.dE) * self.dE
        else:
            E_max_R = math.floor(E_max_p/self.dE) * self.dE

        NE = (E_max_R - E_min_R) / self.dE
        dt = self.time_res

        # Prep arrays
        self.energies = numpy.linspace(E_min_R, E_max_R - self.dE, NE)
        self.spin_up_energy = numpy.zeros(NE)
        self.spin_down_energy = numpy.zeros(NE)

        def energy_to_time(conv, energy):
            return math.sqrt(conv/energy)

        # Rebin data
        for i, E in enumerate(self.energies):
            t_L = energy_to_time(c, E + self.dE / 2)
            t_S = energy_to_time(c, E - self.dE / 2)

            # clamp
            t_L = t_L if t_L <= t_max else t_max
            t_S = t_S if t_S > t_min else t_min

            # with some math we could back calculate the index, but we don't need to
            t_L_idx = numpy.searchsorted(self.timing, t_L)
            t_S_idx = numpy.searchsorted(self.timing, t_S)

            self.spin_up_energy[i] = numpy.sum(self.spin_up_time[t_L_idx:t_S_idx]) + (
                (self.timing[t_L_idx] - t_L) * self.spin_up_time[t_L_idx] +
                (t_S - self.timing[t_L_idx]) * self.spin_up_time[t_S_idx - 1]
            ) / self.dE

            self.spin_down_energy[i] = numpy.sum(self.spin_down_time[t_L_idx:t_S_idx]) + (
                (self.timing[t_L_idx] - t_L) * self.spin_down_time[t_L_idx] +
                (t_S - self.timing[t_L_idx]) * self.spin_down_time[t_S_idx - 1]
            ) / self.dE


class Spectrum3D(Viewable):
    def convert_to_kinetic_energy(self):
        """
        Converts the range on the energy to kinetic energy
        """

        self.offset_axis('E', self.metadata['photon_energy'])
        self.rename_axis('E', 'KE')


    def convert_to_binding_energy(self):
        """
        This function is responsible for two things, it normalizes the
        fermi level to occur at the same index in all energy slices if one of the
        scan degrees of freedom was photon energy, and it also applies an offset
        to the axes so that the fermi level occurs at zero binding energy for each
        slice
        """

        # If one of the axes was photon energy, then we need to shift all the
        # slices, this is because differing incident photon energies can shift
        # the observed fermi level

        angle_axis_name = 'pixel' if 'pixel' in set(self.axis_names) else 'detector_angle'
        angle_axis = self.axis_names.index(angle_axis_name)

        marginal_along_pixels = numpy.sum(self.data, axis=angle_axis)
        nhv, nE = marginal_along_pixels.shape

        energy_axis = self.axis_names.index('E')
        old_energy_bounds = self.axes[energy_axis]['bounds']
        self.original_bounds = copy.deepcopy(self.bounds)
        location_of_fermi_edge = None
        window = self.metadata['data_preparation'].get(
            'energy_reject_window', 50)

        if 'hv' in self.axis_names:
            # we must have one of the axis names equal to 'E' rather than 'BE'
            # otherwise we don't need to be converting!
            hv_axis = self.axis_names.index('hv')

            # for convenience, let's put the energy axis at the end
            if energy_axis < hv_axis:
                marginal_along_pixels = marginal_along_pixels.T



            # Use gradient filtering to perform edge detection and align fermi levels
            # this also goes more simply if we smooth just a little before we apply
            # the gradient
            smoothed_marginal = ndimage.gaussian_filter(marginal_along_pixels, 3)
            edges = ndimage.sobel(smoothed_marginal, axis=1)

            index_values = numpy.argmin(edges[:,window:-window], axis=1) + window

            min_index = numpy.min(index_values)
            new_energy_axis_size = nE - (max(index_values) - min(index_values))
            new_range = nE - (numpy.max(index_values) - numpy.min(index_values))

            for i in range(nhv):
                lslices = {
                    'hv': numpy.s_[i],
                    'E': numpy.s_[0:new_energy_axis_size],
                }
                rslices = {
                    'hv': numpy.s_[i],
                    'E': numpy.s_[index_values[i] - min_index:
                                  index_values[i] - min_index + new_energy_axis_size],
                }

                lslice = [lslices.get(axis, numpy.s_[:]) for axis in self.axis_names]
                rslice = [rslices.get(axis, numpy.s_[:]) for axis in self.axis_names]

                self.data[lslice] = self.data[rslice]

            # toss out junk values, to do this we have to build a weird slice
            # because the energy axis could be located anywhere
            slices = [numpy.s_[0:new_range] if axis_name == 'E' else numpy.s_[:]
                      for axis_name in self.axis_names]
            self.data = self.data[slices]
            self.normalized_fermi_level = True

        else:
            # Because the fermi levels should be lined up along the swept angle,
            # we can sum across this scan degree of freedom before we try to find
            # the edge
            energy_marginal = numpy.sum(marginal_along_pixels, axis=0)
            location_of_fermi_edge = None
            edge = ndimage.sobel(ndimage.gaussian_filter(
                energy_marginal, 3))

            fermi_parameters = self.metadata.get('data_preparation', {}).get(
                'fermi', None)

            if fermi_parameters is not None:
                peak_range = fermi_parameters['peak_range']
                min_index = numpy.argmin(edge[peak_range[0]:peak_range[1]]) + peak_range[0]
            else:
                min_index = numpy.argmin(edge[window:-window]) + window
            new_range = nE

        # adjust bounds on the energy and mark as binding energy
        # min_energy = min_index
        _, __, e_delta = old_energy_bounds
        new_energy_bounds = (
            -(min_index - 1) * e_delta,
            (new_range - min_index) * e_delta,
            e_delta,)

        self.bounds[energy_axis] = new_energy_bounds
        self.axes[energy_axis] = {
            'name': 'BE',
            'bounds': new_energy_bounds,
        }

    def value_to_index(self, value, axis=None):
        if axis == 'pixel':
            # pixel values are already indices
            return value

        return super(Spectrum3D, self).value_to_index(value, axis)

    @property
    def is_viewable(self):
        """
        We should not allow constructing k-space 'View's of a spectrum if it has
        not been converted to binding energy, if one of the axes still represents
        pixels, or if one of the axes was photon energy and the energy levels have
        not been shifted appropriately.
        """

        if 'hv' in set(self.axis_names) and not self.normalized_fermi_level:
            return False

        # don't allow viewing if the energy has not been converted to binding energy
        if 'E' in set(self.axis_names):
            return False

        # definitely don't allow creating views if we are still in pixel-space
        if 'pixel' in set(self.axis_names):
            warnings.warn('Be sure to convert from pixel-space to angle-space ' +
                          'before attempting any k-space conversions!')
            return False

        return True

    def set_normal_incidence_pixel(self, normal_incidence=None):
        normal_incidence = self.metadata['data_preparation'].get(
            'normal_incidence_pixel', normal_incidence)

        if normal_incidence is None:
            # instead of setting the normal incidence location and swapping pixels
            # for angle, instead we're going to prompt the user for the appropriate
            # pixel

            print('Set the normal incidence pixel in order to finish preliminary ' +
                  'data preparation')
            self.tslice(make_plot=True, BE=0)
            return

        deg_per_pixel = self.metadata['data_preparation']['degrees_per_pixel']
        rad_per_pixel = deg_per_pixel * consts.RAD_PER_DEG

        pixel_axis_index = self.axis_names.index('pixel')
        pixel_bounds = self.bounds[pixel_axis_index]

        n_pixels = self.data.shape[pixel_axis_index]

        # provide new bounds in *radians*
        new_bounds = (
            - (normal_incidence - 1) * rad_per_pixel,
            (n_pixels - normal_incidence) * rad_per_pixel,
            rad_per_pixel
        )

        self.bounds[pixel_axis_index] = new_bounds
        self.axes[pixel_axis_index] = {
            'name': 'detector_angle',
            'bounds': new_bounds,
        }

    def clean_spectrum(self):
        """
        'clean_spectrum' is a hook for a variety of methods that we use in order
        to remove artifacts from the detector, and to correct for non-linearities.

        At the moment we only do artifact removal, but my plan is to allow a key
        in the cleaning dict of the dataset file to indicate a detector non-linearity
        curve.
        """

        self.was_spectrum_cleaned = False
        data_prep_parameters = self.metadata.get('data_preparation', {})
        try:
            cleaning_parameters = data_prep_parameters['cleaning']
        except KeyError:
            return

        if cleaning_parameters.get('skip', False):
            return

        # at this point, the axis is still in the form of kinetic energy, so
        # the axis name will be 'E' rather than 'BE'
        # sum along pixel, E slices
        keep_axes = {'pixel', 'E'}
        sum_axis_index = None
        for i, a in enumerate(self.axis_names):
            if a not in keep_axes:
                sum_axis_index = i

        if sum_axis_index is None:
            raise AnalysisError('Could not clean spectrum because no axis to sum ' +
                                'over could be found.')

        pixel_axis_index = self.axis_names.index('pixel')
        n_pixels = self.data.shape[pixel_axis_index]
        pixel_range = cleaning_parameters.get('pixel_range', [0, n_pixels])


        windowed_integrated_image = numpy.sum(
            self.tslice(pixel=pixel_range), axis=sum_axis_index)


        self._integrated_image = windowed_integrated_image

        # need to shift axes potentially, we do this stuff a lot, might be nice
        # to have a utility
        integrated_pixel_axis_index = pixel_axis_index
        if pixel_axis_index > sum_axis_index:
            integrated_pixel_axis_index -= 1

        self._filter_masks = {}

        # build all the appropriate masks to clean bad pixels
        filter_modes = cleaning_parameters.get('filters', {})
        for f, params in filter_modes.items():
            if f == 'sobel':
                # Use Sobel filtering in order to find bad pixels, this means we
                # want to filter along the direction of constant E
                r = (ndimage.sobel(
                    self._integrated_image, axis=integrated_pixel_axis_index)
                     / self._integrated_image)
                max_i = params.get('max_relative_intensity', 0.1) * numpy.max(r)
                r[r < max_i] = 0
                r[r >= max_i] = 1
                self._filter_masks['sobel'] = numpy.roll(r, 1, axis=0)
                self._filter_masks['sobel'][0] = 0
            if f == 'median':
                # Use median filtering, this can provide complementary
                # information to the Sobel filtering because
                q = (ndimage.median_filter(self._integrated_image, size=(2, 2))
                     / self._integrated_image)
                qq = (self._integrated_image /
                      ndimage.median_filter(self._integrated_image, size=(2, 2)))
                q = q - numpy.average(q)
                qq = qq - numpy.average(qq)
                max_i = params.get('max_relative_intensity', 0.024) * numpy.max(q)
                max_i_qq = params.get('max_relative_intensity', 0.024) * numpy.max(qq)
                q[numpy.abs(q) >= max_i] = 1
                q[numpy.abs(q) < max_i] = 0
                qq[numpy.abs(qq) >= max_i_qq] = 1
                qq[numpy.abs(qq) < max_i_qq] = 0
                self._filter_masks['median_low'] = numpy.roll(q, -1, axis=0)
                # Fix positive masks that rolled off the bottom
                self._filter_masks['median_low'][0] = 0

                # repeat for the second filter
                self._filter_masks['median_high'] = numpy.roll(qq, -1, axis=0)
                self._filter_masks['median_high'][0] = 0


        # We need to temporarily put the integrated axis at the end of the array,
        # as it makes the process of taking the filter using broadcasting much
        # simpler
        d = numpy.moveaxis(self.data, sum_axis_index, -1)

        self._mask = numpy.zeros(d.shape[:-1], dtype=bool)
        mask_slices = [numpy.s_[:], numpy.s_[pixel_range[0]:pixel_range[1]]]

        # swap the slices around if the integrated axis wasn't first
        # WARNING, this assumes a three dimensional spectrum
        if integrated_pixel_axis_index == 0:
            mask_slices = mask_slices[::-1]

        # Now that we've computed individual masks, we need to OR them together
        self._mask[mask_slices] = numpy.logical_or.reduce(
            [m.astype(bool) for m in self._filter_masks.values()])

        # Next we can build the smoothed pixels and mask them into the array
        nearest_neighbors = numpy.transpose(numpy.array([[[0,1,0],
                                                          [1,0,1],
                                                          [0,1,0]]]))

        d[self._mask] = float('nan')

        with warnings.catch_warnings():
            # Ignore varnings of NaN values
            warnings.simplefilter('ignore', RuntimeWarning)

            qq = ndimage.generic_filter(
                d, numpy.nanmean,
                footprint=nearest_neighbors, mode='constant', cval=0)[self._mask]

        d[self._mask] = qq

        # as a last resort, patch in mean values
        d[numpy.isnan(d)] = numpy.nanmean(d)

        # commit to new data
        self.data = numpy.moveaxis(d, -1, sum_axis_index)


    def convert_analyzer_sweep_angle(self):
        try:
            analyzer_sweep_angle_axis = self.axis_names.index('detector_sweep_angle')
        except ValueError:
            # not one of the angles, so we don't have to bother
            return

        angle_bounds = self.bounds[analyzer_sweep_angle_axis]
        offset = self.metadata['data_preparation'].get('sweep_angle_offset', 0)
        new_angle_bounds = (
            angle_bounds[0] * consts.RAD_PER_DEG + offset,
            angle_bounds[1] * consts.RAD_PER_DEG + offset,
            angle_bounds[2] * consts.RAD_PER_DEG,
        )

        self.bounds[analyzer_sweep_angle_axis] = new_angle_bounds
        self.axes[analyzer_sweep_angle_axis]['bounds'] = new_angle_bounds

    def __init__(self, data=None, axes=None, bounds=None, metadata=None):
        super(Spectrum3D, self).__init__(data=data, axes=axes, bounds=bounds)

        self.metadata = metadata
        self.views = {} # a cache for constructed views

        self.clean_spectrum()

        # flags for various data preparation steps and to control allowable
        # behaviors
        self.normal_incidence_angle_set = False
        self.normalized_fermi_level = False

        # if one of the axis directions included photon energy,
        # we need to shift all the energies so that the binding energy
        # is set to zero at the fermi level of each constant hv slice
        # this amounts to taking marginals along the detector acceptance
        # angle and fitting each constant hv curve to a Fermi-Dirac
        # distribution.
        target_energy_type = metadata.get('target_energy_type', 'BE')
        if target_energy_type == 'BE':
            self.convert_to_binding_energy()
        elif target_energy_type == 'KE':
            self.convert_to_kinetic_energy()
        elif target_energy_type == 'E':
            pass
        else:
            raise AnalysisError('Unknown energy type {}'.format(target_energy_type))

        # we also need to prompt to find the angle of normal incidence and set
        # this to zero, if it's provided in the metadata, then we're good,
        # otherwise we should bug about it
        self.set_normal_incidence_pixel()

        # If one of the scan degrees of freedom was the analyzer sweep angle,
        # then we need to convert this axis to radians and add an optional
        # offset from the data preparation section of the dataset
        self.convert_analyzer_sweep_angle()
