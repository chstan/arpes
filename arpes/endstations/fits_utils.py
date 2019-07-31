import numpy as np
from ast import literal_eval
import functools
from collections import Iterable

from arpes.utilities.funcutils import iter_leaves, collect_leaves

__all__ = ('extract_coords', 'find_clean_coords',)

DEFAULT_DIMENSION_RENAMINGS = {
    'Beta': 'beta',
    'Theta': 'theta',
    'Delay': 'delay',
    'Sample-X': 'cycle',
    'null': 'cycle',
    'Mira': 'pump_power',
    'X': 'x',
    'Y': 'y',
    'Z': 'z',
}


def extract_coords(attrs, dimension_renamings=None):
    """
    Does the hard work of extracting coordinates from the scan description.
    :param attrs:
    :param dimension_renamings:
    :return:
    """
    if dimension_renamings is None:
        dimension_renamings = DEFAULT_DIMENSION_RENAMINGS

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

                name = dimension_renamings.get(name, name)

                scan_dimension.append(name)
                scan_shape.append(n)
                scan_coords[name] = np.linspace(start, end, n, endpoint=True)
        else:  # tabulated scan, this is more complicated
            name, n = (
                attrs['NM_%g_0' % loop],
                attrs['NMPOS_%g' % loop],
            )

            try:
                n_regions_key = {'Delay': 'DS_NR'}.get(name, 'DS_NR')
                n_regions = attrs[n_regions_key]

                name = dimension_renamings.get(name, name)
            except KeyError:
                if 'ST_{}_1'.format(loop) in attrs:
                    assert(False and "More than one region detected but unhandled.")

                n_regions = 1
                name = dimension_renamings.get(name, name)

            coord = np.array(())
            for region in range(n_regions):
                start, end, n = (
                    attrs['ST_%g_%g' % (loop, region,)],
                    attrs['EN_%g_%g' % (loop, region,)],
                    attrs['N_%g_%g' % (loop, region,)],
                )

                coord = np.concatenate((coord, np.linspace(
                    start, end, n, endpoint=True),))

            scan_dimension.append(name)
            scan_shape.append(len(coord))
            scan_coords[name] = coord
    return scan_coords, scan_dimension, scan_shape


def find_clean_coords(hdu, attrs, spectra=None, mode='ToF', dimension_renamings=None):
    """
    Determines the scan degrees of freedom, the shape of the actual "spectrum"
    and reads and parses the coordinates from the header information in the recorded
    scan.

    Note: because different scan configurations can have different values of the detector coordinates, such as
    for instance when you record in two different angular modes of the spectrometer or when you record XPS spectra
    at different binding energies, we need to be able to provide separate coordinates for each of the different scans.

    In the case where there is a unique coordinate, we will return only that coordinate, under the anticipated name,
    such as 'eV'. Otherwise, we will return the coordinates that different between the scan configurations under the
    spectrum name, and with unique names, such as 'eV-Swept_Spectra0'.

    TODO Write data loading tests to ensure we don't break MC compatibility

    :param spectra:
    :param hdu:
    :param attrs:
    :param mode: Available modes are "ToF", "MC". This customizes the read process
    :param dimension_renamings:
    :return: (coordinates, dimensions, np shape of actual spectrum)
    """
    if dimension_renamings is None:
        dimension_renamings = DEFAULT_DIMENSION_RENAMINGS

    scan_coords, scan_dimension, scan_shape = extract_coords(attrs, dimension_renamings=dimension_renamings)

    # bit of a hack to deal with the internal motor used for the swept spectra being considered as a cycle
    if 'cycle' in scan_coords and len(scan_coords['cycle']) > 200:
        idx = scan_dimension.index('cycle')

        real_data_for_cycle = hdu.data.columns['null'].array

        scan_coords['cycle'] = real_data_for_cycle
        scan_shape[idx] = len(real_data_for_cycle)

    scan_dimension = [dimension_renamings.get(s, s) for s in scan_dimension[::-1]]
    scan_coords = {dimension_renamings.get(k, k): v for k, v in scan_coords.items()}
    extra_coords = {}
    scan_shape = scan_shape[::-1]

    spectrum_shapes = {}
    dimensions_for_spectra = {}

    if spectra is None:
        spectra = hdu.columns.names

    if isinstance(spectra, str):
        spectra = [spectra]

    for spectrum_key in spectra:
        skip_names = {
            lambda name: True if ('beamview' in name or 'IMAQdx' in name) else False,
        }

        if spectrum_key is None:
            spectrum_key = hdu.columns.names[-1]

        if isinstance(spectrum_key, str):
            spectrum_key = hdu.columns.names.index(spectrum_key) + 1

        spectrum_name = hdu.columns.names[spectrum_key - 1]
        loaded_shape_from_header = False
        desc = None

        should_skip = False
        for skipped in skip_names:
            if callable(skipped) and skipped(spectrum_name):
                should_skip = True
            elif skipped == spectrum_name:
                should_skip = True
        if should_skip:
            continue

        try:
            offset = hdu.header['TRVAL%g' % spectrum_key]
            delta = hdu.header['TDELT%g' % spectrum_key]
            offset = literal_eval(offset) if isinstance(offset, str) else offset
            delta = literal_eval(delta) if isinstance(delta, str) else delta

            try:
                shape = hdu.header['TDIM%g' % spectrum_key]
                shape = literal_eval(shape) if isinstance(shape, str) else shape
                loaded_shape_from_header = True
            except:
                shape = hdu.data.field(spectrum_key - 1).shape

            try:
                desc = hdu.header['TDESC%g' % spectrum_key]
                if '(' in desc:
                    # might be a malformed tuple, we can't use literal_eval unfortunately
                    desc = desc.replace('(', '').replace(')', '').split(',')

                if isinstance(desc, str):
                    desc = (desc,)
            except KeyError:
                pass

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

        if mode == 'ToF':
            rest_shape = shape[len(scan_shape):]
        else:
            if isinstance(desc, tuple):
                rest_shape = shape[-len(desc):]
            elif not loaded_shape_from_header:
                rest_shape = shape[1:]
            else:
                rest_shape = shape

        assert(len(offset) == len(delta) and len(delta) == len(rest_shape))

        # Build the actually coordinates
        coords = [np.linspace(o, o + s * d, s, endpoint=False)
                  for o, d, s in zip(offset, delta, rest_shape)]

        # We need to do smarter inference here
        def infer_hemisphere_dimensions():
            # scans can be two dimensional per frame, or a
            # scan can be either E or K integrated, or something I've never seen before
            # try to get the description or the UNIT
            if desc is not None:
                RECOGNIZED_DESCRIPTIONS = {
                    'eV': 'eV',
                    'pixels': 'pixel',
                    'pixel': 'pixel',
                }

                if all(d in RECOGNIZED_DESCRIPTIONS for d in desc):
                    return [RECOGNIZED_DESCRIPTIONS[d] for d in desc]

            try:
                # TODO read above like desc
                unit = hdu.header['TUNIT{}'.format(spectrum_key)]
                RECOGNIZED_UNITS = {
                    # it's probably 'arb' which doesn't tell us anything...
                    # because all spectra have arbitrary absolute intensity
                }
                if all(u in RECOGNIZED_UNITS for u in unit):
                    return [RECOGNIZED_UNITS[u] for u in unit]
            except KeyError:
                pass

            # Need to fall back on some human in the loop to improve the read
            # here
            import pdb
            pdb.set_trace()


        # TODO for cleanup in future, these should be provided by the implementing endstation class, so they do not
        # get so cluttered, best way will be to make this function a class method, and use class attributes for
        # each of `coord_names_for_spectrum`, etc. For now, patching to avoid error with the microscope camera images
        # at BL7
        coord_names_for_spectrum = {
            'Time_Spectra': ['time'],
            'Energy_Spectra': ['eV'],
            # MC hemisphere image, this can still be k-integrated, E-integrated, etc
            'wave':  ['time'],
            'targetPlus': ['time'],
            'targetMinus': ['time'],
            'Energy_Target_Up': ['eV'],
            'Energy_Target_Down': ['eV'],
            'Energy_Up': ['eV'],
            'Energy_Down': ['eV'],
            'Energy_Pol': ['eV'],
        }

        spectra_types = {
            'Fixed_Spectra',
            'Swept_Spectra',
        }

        time_spectra_type = {
            'Time_Target',
        }
        coord_names = None
        if spectrum_name not in coord_names_for_spectrum:
            # Don't remember what the MC ones were, so I will wait to do those again
            # Might have to add new items for new spectrometers as well
            if any(s in spectrum_name for s in spectra_types):
                coord_names = infer_hemisphere_dimensions
            elif any(s in spectrum_name for s in time_spectra_type):
                coord_names = ['time',]
            else:
                import pdb
                pdb.set_trace()
        else:
            coord_names = coord_names_for_spectrum[spectrum_name]

        if callable(coord_names):
            coord_names = coord_names()
            if len(coord_names) > 1 and mode == 'MC':
                # for whatever reason, the main chamber records data
                # in nonstandard byte order
                coord_names = coord_names[::-1]
                rest_shape = list(rest_shape)[::-1]
                coords = coords[::-1]

        coords_for_spectrum = dict(zip(coord_names, coords))
        # we need to store the coordinates that were kept in a table separately, because they are allowed to differ
        # between different scan configurations in the same file
        if mode == 'ToF':
            extra_coords.update(coords_for_spectrum)
        else:
            extra_coords[spectrum_name] = coords_for_spectrum
        dimensions_for_spectra[spectrum_name] = \
            tuple(scan_dimension) + tuple(coord_names)
        spectrum_shapes[spectrum_name] = tuple(scan_shape) + tuple(rest_shape)
        coords_for_spectrum.update(scan_coords)

    extra_coords.update(scan_coords)

    if mode != 'ToF':
        detector_coord_names = [k for k, v in extra_coords.items() if isinstance(v, dict)]

        from collections import Counter
        c = Counter(item for name in detector_coord_names for item in extra_coords[name])
        conflicted = [k for k, v in c.items() if v != 1 and k != 'cycle']

        flat_coordinates = collect_leaves(extra_coords)
        def can_resolve_conflict(c):
            coordinates = flat_coordinates[c]

            if not isinstance(coordinates, list) or len(coordinates) < 2:
                return True

            # check if list of arrays is all equal
            return functools.reduce(lambda x, y: (np.array_equal(x[1], y) and x[0], y), coordinates, (True, coordinates[0]))[0]

        conflicted = [c for c in conflicted if not can_resolve_conflict(c)]

        def clarify_dimensions(dims, sname):
            return [d if d not in conflicted else d + '-' + sname for d in dims]

        def clarify_coordinate(coordinates, sname):
            if not isinstance(coordinates, dict):
                return coordinates

            return {k if k not in conflicted else k + '-' + sname: v for k, v in coordinates.items()}

        dimensions_for_spectra = {k: clarify_dimensions(v, k) for k, v in dimensions_for_spectra.items()}
        extra_coords = {k: clarify_coordinate(v, k) for k, v in extra_coords.items()}
        extra_coords = dict(iter_leaves(extra_coords))

    return extra_coords, dimensions_for_spectra, spectrum_shapes
