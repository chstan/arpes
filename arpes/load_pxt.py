from pathlib import Path
from arpes.utilities import rename_keys
import numpy as np
import re

import xarray as xr

__all__ = ('pxt_to_hdf', 'read_single_pxt', 'read_separated_pxt',)

binary_header_bytes = 10

igor_wave_header_dtype = np.dtype([
    ('next', '>u8',),
    ('creation_date', '>u8'),
    ('mod_date', '>u8'),
    ('type', '>H'),
    ('d_lock', '>H'),
    ('wave_header_pad1', 'B', (62,),), # unused
    ('n_points', '>u8'),
    ('wave_header_version', '>u4'),
    ('spacer', '>u8'),
    ('wave_name', 'S', 32),
    ('wave_header_pad2', 'B', (8,)),
    ('dim_sizes', '>u4', (4,)),
    ('dim_scales', '>d', (4,)),
    ('dim_offsets', '>d', (4,)),
    ('data_units', 'b', (4,),),
    ('dim_units', ('S', 4), (4,)),
    ('final_spacer', 'b', 2 + 8 * 18),
])

def read_igor_binary_wave(raw_bytes):
    """
    Some weirdness here with the byte ordering and the target datatype.
    Data loading might not be perfectly correct as a result, maybe don't trust
    for now.

    :param raw_bytes:
    :return:
    """
    header = np.fromstring(raw_bytes[:igor_wave_header_dtype.itemsize],
                           dtype=igor_wave_header_dtype).item()

    point_size = 8
    n_points = header[6]
    wave_name = header[9]
    dim_sizes, dim_scales, dim_offsets = header[11], header[12], header[13]
    data_units = header[14]
    dim_units = header[15]

    # approximate offsets
    # >i8 -1
    # <i8 0
    # >u8 -1
    # <u8 0
    offset = -1

    wave_data = np.fromstring(
        raw_bytes[igor_wave_header_dtype.itemsize + offset:igor_wave_header_dtype.itemsize + n_points * point_size + offset],
        dtype='>u8')

    wave_data = wave_data / np.max(wave_data)
    wave_data[wave_data == 1] = 0 # some weird bit shifting going on?

    names_from_units = {
        'eV': 'eV',
        'deg': 'phi',
    }

    dim_sizes = [d for d in dim_sizes if d]
    n_dims = len(dim_sizes)

    # please pylint forgive me
    dims = [(names_from_units.get(dim_units[i].decode('ascii'), dim_units[i].decode('ascii')),
             np.linspace(dim_offsets[i], dim_offsets[i] + (dim_sizes[i] - 1) * dim_scales[i],
                         dim_sizes[i])) for i in range(n_dims)]
    coords = dict(dims)

    return xr.DataArray(
        wave_data.reshape(dim_sizes[::-1]),
        coords=coords,
        dims=[d[0] for d in dims][::-1],
    )


def read_header(header_bytes: bytes):
    header_as_string = header_bytes.decode('utf-8')
    lines = [x for x in header_as_string.replace('\r', '\n').split('\n') if x]
    lines = [x for x in lines if '=' in x]

    header = {}
    for l in lines:
        fragments = l.split('=')
        first, rest = fragments[0], '='.join(fragments[1:])

        try:
            rest = int(rest)
        except ValueError:
            try:
                rest = float(rest)
            except ValueError:
                pass

        header[first.lower().replace(' ', '_')] = rest

    return rename_keys(header, {
        'sample_x': 'x',
        'sample_y_(vert)': 'y',
        'sample_y': 'y',
        'sample_z': 'z',
        'bl_energy': 'hv',
    })

def read_single_pxt(reference_path: Path, separator=b'[BCS]'):
    bytes_for_file = reference_path.read_bytes()

    content, header = bytes_for_file.split(separator)
    header = read_header(header)
    wave = read_igor_binary_wave(content[10:])
    wave.attrs.update(header)
    return wave


def read_separated_pxt(reference_path: Path, separator=b'[BCS]'):
    # determine if separated or not
    name_match = re.match(r'([\w+]+)S[0-9][0-9][0-9]\.pxt', reference_path.name)

    if name_match is None:
        return read_single_pxt(reference_path)

    # otherwise need to collect all of the components
    fragment = name_match.groups()[0]
    components = list(reference_path.parent.glob('{}S*.pxt'.format(fragment)))
    components.sort()

    frames = [read_single_pxt(f) for f in components]

    scan_coords = ['hv', 'polar', 'timed_power', 'tilt',]

    scan_coord = None
    max_different_values = -np.inf
    for possible_scan_coord in scan_coords:
        coordinates = [f.attrs[possible_scan_coord] for f in frames]
        n_different_values = len(set(coordinates))
        if n_different_values > max_different_values:
            max_different_values = n_different_values
            scan_coord = possible_scan_coord

    for f in frames:
        f.coords[scan_coord] = f.attrs[scan_coord]

    frames.sort(key=lambda x: x.coords[scan_coord])
    return xr.concat(frames, scan_coord)

def pxt_to_hdf(reference_path: Path, separator=b'[BCS]'):
    bytes_for_file = reference_path.read_bytes()

    content, header = bytes_for_file.split(separator)
    return content, read_header(header)