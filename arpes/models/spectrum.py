# TODO Reorganize code in here. You should be able to register new loading code in order to
# make this code more portable to other beamlines and spectrometers.

import copy
import itertools
import os.path
import subprocess
import warnings
from ast import literal_eval
from collections import Iterable

import h5py
import numpy
import xarray
from astropy.io import fits

import arpes.config
from arpes.preparation import infer_center_pixel, replace_coords
from arpes.preparation.tof_preparation import convert_SToF_to_energy
from arpes.provenance import provenance_from_file
from arpes.utilities import rename_keys

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


def load_BL10(metadata: dict=None, filename: str=None):
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

    coords, dimensions, spectrum_shape = find_clean_coords(hdu, metadata)
    columns = hdu.columns
    dataset = None

    column_renamings = {}
    take_columns = columns

    spectra_names = [name for name in take_columns if name in columns.names]

    skip_frags = {}
    skip_predicates = {lambda k: any(s in k for s in skip_frags)}
    metadata = {k: v for k, v in metadata.items()
                if not any(pred(k) for pred in skip_predicates)}

    data_vars = {k: (dimensions[k], hdu.data[k].reshape(spectrum_shape[k]), metadata)
                 for k in spectra_names}
    data_vars = rename_keys(data_vars, column_renamings)

    hdulist.close()

    relevant_dimensions = {k for k in coords.keys() if k in
                           set(itertools.chain(*[l[0] for l in data_vars.values()]))}
    relevant_coords = {k: v for k, v in coords.items() if k in relevant_dimensions}

    dataset = xarray.Dataset(
        data_vars,
        relevant_coords,
        metadata,
    )

    provenance_from_file(dataset, data_loc, {
        'what': 'Loaded Spin-ToF dataset',
        'by': 'load_DLD',
    })

    return dataset


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
