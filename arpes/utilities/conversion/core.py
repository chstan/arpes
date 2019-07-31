"""
Helper functions for coordinate transformations. All the functions here
assume standard polar angles, as given in the guide: https://arpes.netlify.com/#/spectra


Functions here must accept constants or numpy arrays as valid inputs,
so all standard math functions have been replaced by their equivalents out
of numpy. Array broadcasting should handle any issues or weirdnesses that
would encourage the use of direct iteration, but in case you need to write
a conversion directly, be aware that any functions here must work on arrays
as well for consistency with client code.

Everywhere:

Kinetic energy -> 'kinetic_energy'
Binding energy -> 'eV', for convenience (negative below 0)
Photon energy -> 'hv'
"""

# pylint: disable=W0613, C0103

import collections
import warnings
from copy import deepcopy

import numpy as np
import scipy.interpolate
import xarray as xr

from arpes.provenance import provenance, update_provenance
from arpes.exceptions import AnalysisError
from arpes.utilities import normalize_to_spectrum
from .kx_ky_conversion import *
from .kz_conversion import *

__all__ = ['convert_to_kspace', 'slice_along_path']

# TODO Add conversion utilities that work for lower dimensionality, i.e. the ToF
# TODO Check if conversion utilities work for constant energy cuts


def infer_kspace_coordinate_transform(arr: xr.DataArray):
    """
    Infers appropriate coordinate transform for arr to momentum space.

    This takes into account the extra metadata attached to arr that might be
    useful in inferring the requirements of the coordinate transform, like the
    orientation of the spectrometer slit, and other experimental concerns
    :param arr:
    :return: dict with keys ``target_coordinates``, and a map of the appropriate
    conversion functions
    """
    old_coords = deepcopy(list(arr.coords))
    assert ('eV' in old_coords)
    old_coords.remove('eV')
    old_coords.sort()

    new_coords = {
        ('phi',): ['kp'],
        ('beta', 'phi',): ['kx', 'ky'],
        ('phi', 'theta',): ['kx', 'ky'],
        ('phi', 'psi',): ['kx', 'ky'],
        ('hv', 'phi',): ['kp', 'kz'],
        ('beta', 'hv', 'phi',): ['kx', 'ky', 'kz'],
        ('hv', 'phi', 'theta',): ['kx', 'ky', 'kz'],
        ('hv', 'phi', 'psi',): ['kx', 'ky', 'kz'],
    }.get(tuple(old_coords))

    # At this point we need to do a bit of work in order to determine the functions
    # that interpolate from k-space back into the recorded variable space

    # TODO Also provide the Jacobian of the coordinate transform to properly
    return {
        'dims': new_coords,
        'transforms': {

        },
        'calculate_bounds': None,
        'jacobian': None,
    }


def grid_interpolator_from_dataarray(arr: xr.DataArray, fill_value=0.0, method='linear',
                                     bounds_error=False):
    """
    Translates the contents of an xarray.DataArray into a scipy.interpolate.RegularGridInterpolator.

    This is principally used for coordinate translations.
    """
    flip_axes = set()
    for d in arr.dims:
        c = arr.coords[d]
        if len(c) > 1 and c[1] - c[0] < 0:
            flip_axes.add(d)

    values = arr.values
    for dim in flip_axes:
        values = np.flip(values, arr.dims.index(dim))

    return scipy.interpolate.RegularGridInterpolator(
        points=[arr.coords[d].values[::-1] if d in flip_axes else arr.coords[d].values for d in arr.dims],
        values=values,
        bounds_error=bounds_error, fill_value=fill_value, method=method)


def slice_along_path(arr: xr.DataArray, interpolation_points=None, axis_name=None, resolution=None,
                     shift_gamma=True, extend_to_edge=False, **kwargs):
    """
    TODO: There might be a little bug here where the last coordinate has a value of 0, causing the interpolation to loop
    back to the start point. For now I will just deal with this in client code where I see it until I understand if it is
    universal.

    Interpolates along a path through a volume. If the volume is higher dimensional than the desired path, the
    interpolation is broadcasted along the free dimensions. This allows one to specify a k-space path and receive
    the band structure along this path in k-space.

    Points can either by specified by coordinates, or by reference to symmetry points, should they exist in the source
    array. These symmetry points are translated to regular coordinates immediately, but are provided as a convenience.
    If not all points specify the same set of coordinates, an attempt will be made to unify the coordinates. As an example,
    if the specified path is (kx=0, ky=0, T=20) -> (kx=1, ky=1), the path will be made between (kx=0, ky=0, T=20) ->
    (kx=1, ky=1, T=20). On the other hand, the path (kx=0, ky=0, T=20) -> (kx=1, ky=1, T=40) -> (kx=0, ky=1) will result
    in an error because there is no way to break the ambiguity on the temperature for the last coordinate.

    A reasonable value will be chosen for the resolution, near the maximum resolution of any of the interpolated
    axes by default.

    This function transparently handles the entire path. An alternate approach would be to convert each segment
    separately and concatenate the interpolated axis with xarray.

    If the sentinel value 'G' for the Gamma point is included in the interpolation points, the coordinate axis of the
    interpolated coordinate will be shifted so that its value at the Gamma point is 0. You can opt out of this with the
    parameter 'shift_gamma'

    :param arr: Source data
    :param interpolation_points: Path vertices
    :param axis_name: Label for the interpolated axis. Under special circumstances a reasonable name will be chosen,
    such as when the interpolation dimensions are kx and ky: in this case the interpolated dimension will be labeled kp.
    In mixed or ambiguous situations the axis will be labeled by the default value 'inter'.
    :param resolution: Requested resolution along the interpolated axis.
    :param shift_gamma: Controls whether the interpolated axis is shifted to a value of 0 at Gamma.
    :param extend_to_edge: Controls whether or not to scale the vector S - G for symmetry point S so that you interpolate
    to the edge of the available data
    :param kwargs:
    :return: xr.DataArray containing the interpolated data.
    """

    if interpolation_points is None:
        raise ValueError('You must provide points specifying an interpolation path')

    def extract_symmetry_point(name):
        raw_point = arr.attrs['symmetry_points'][name]
        G = arr.attrs['symmetry_points']['G']

        if not extend_to_edge or name == 'G':
            return raw_point

        # scale the point so that it reaches the edge of the dataset
        S = np.array([raw_point[d] for d in arr.dims if d in raw_point])
        G = np.array([G[d] for d in arr.dims if d in raw_point])

        scale_factor = np.inf
        for i, d in enumerate([d for d in arr.dims if d in raw_point]):
            dS = (S - G)[i]
            coord = arr.coords[d]

            if np.abs(dS) < 0.001:
                continue

            if dS < 0:
                required_scale = (np.min(coord) - G[i]) / dS
                if required_scale < scale_factor:
                    scale_factor = float(required_scale)
            else:
                required_scale = (np.max(coord) - G[i]) / dS
                if required_scale < scale_factor:
                    scale_factor = float(required_scale)

        S = (S - G) * scale_factor + G
        return dict(zip([d for d in arr.dims if d in raw_point], S))


    parsed_interpolation_points = [
        x if isinstance(x, collections.Iterable) and not isinstance(x, str) else extract_symmetry_point(x)
        for x in interpolation_points
    ]

    free_coordinates = list(arr.dims)
    seen_coordinates = collections.defaultdict(set)
    for point in parsed_interpolation_points:
        for coord, value in point.items():
            seen_coordinates[coord].add(value)
            if coord in free_coordinates:
                free_coordinates.remove(coord)

    for point in parsed_interpolation_points:
        for coord, values in seen_coordinates.items():
            if coord not in point:
                if len(values) != 1:
                    raise ValueError('Ambiguous interpolation waypoint broadcast at dimension {}'.format(coord))
                else:
                    point[coord] = list(values)[0]

    if axis_name is None:
        axis_name = {
            ('beta', 'phi',): 'angle',
            ('chi', 'phi',): 'angle',
            ('phi', 'psi',): 'angle',
            ('phi', 'theta',): 'angle',
            ('kx', 'ky',): 'kp',
            ('kx', 'kz',): 'k',
            ('ky', 'kz',): 'k',
            ('kx', 'ky', 'kz',): 'k'
        }.get(tuple(sorted(seen_coordinates.keys())), 'inter')

        if axis_name == 'angle' or axis_name == 'inter':
            warnings.warn('Interpolating along axes with different dimensions '
                          'will not include Jacobian correction factor.')

    converted_coordinates = None
    converted_dims = free_coordinates + [axis_name]

    path_segments = list(zip(parsed_interpolation_points, parsed_interpolation_points[1:]))

    def element_distance(waypoint_a, waypoint_b):
        delta = np.array([waypoint_a[k] - waypoint_b[k] for k in waypoint_a.keys()])
        return np.linalg.norm(delta)

    def required_sampling_density(waypoint_a, waypoint_b):
        ks = waypoint_a.keys()
        dist = element_distance(waypoint_a, waypoint_b)
        delta = np.array([waypoint_a[k] - waypoint_b[k] for k in ks])
        delta_idx = [abs(d / (arr.coords[k][1] - arr.coords[k][0])) for d, k in zip(delta, ks)]
        return dist / np.max(delta_idx)

    # Approximate how many points we should use
    segment_lengths = [element_distance(*segment) for segment in path_segments]
    path_length = sum(segment_lengths)

    gamma_offset = 0 # offset the gamma point to a k coordinate of 0 if possible
    if 'G' in interpolation_points and shift_gamma:
        gamma_offset = sum(segment_lengths[0:interpolation_points.index('G')])

    if resolution is None:
        resolution = np.min([required_sampling_density(*segment) for segment in path_segments])

    def converter_for_coordinate_name(name):
        def raw_interpolator(*coordinates):
            return coordinates[free_coordinates.index(name)]

        if name in free_coordinates:
            return raw_interpolator

        # Conversion involves the interpolated coordinates
        def interpolated_coordinate_to_raw(*coordinates):
            # Coordinate order is [*free_coordinates, interpolated]
            interpolated = coordinates[len(free_coordinates)] + gamma_offset

            # Start with empty array that we will mask writes onto
            # We need to go with a masking approach rather than a concatenation based one because the coordinates
            # come from np.meshgrid
            dest_coordinate = np.zeros(shape=interpolated.shape)

            start = 0
            for i, l in enumerate(segment_lengths):
                end = start + l
                normalized = (interpolated - start) / l
                seg_start, seg_end = path_segments[i]
                dim_start, dim_end = seg_start[name], seg_end[name]
                mask = np.logical_and(normalized >= 0, normalized < 1)
                dest_coordinate[mask] = \
                    dim_start * (1 - normalized[mask]) + dim_end * normalized[mask]
                start = end

            return dest_coordinate

        return interpolated_coordinate_to_raw

    converted_coordinates = {d: arr.coords[d].values for d in free_coordinates}

    # Adjust this coordinate under special circumstances
    converted_coordinates[axis_name] = np.linspace(0, path_length, int(path_length / resolution)) - gamma_offset

    converted_ds = convert_coordinates(
        arr,
        converted_coordinates,
        {
            'dims': converted_dims,
            'transforms': dict(zip(arr.dims, [converter_for_coordinate_name(d) for d in arr.dims]))
        },
        as_dataset=True
    )

    if axis_name in arr.dims and len(parsed_interpolation_points) == 2:
        if parsed_interpolation_points[1][axis_name] < parsed_interpolation_points[0][axis_name]:
            # swap the sign on this axis as a convenience to the caller
            converted_ds.coords[axis_name].data = -converted_ds.coords[axis_name].data

    if 'id' in converted_ds.attrs:
        del converted_ds.attrs['id']
        provenance(converted_ds, arr, {
            'what': 'Slice along path',
            'by': 'slice_along_path',
            'parsed_interpolation_points': parsed_interpolation_points,
            'interpolation_points': interpolation_points,
        })

    return converted_ds


@update_provenance('Automatically k-space converted')
def convert_to_kspace(arr: xr.DataArray, forward=False, resolution=None, **kwargs):
    """
    "Forward" or "backward" converts the data to momentum space.

    "Backward"
    The standard method. Works in generality by regridding the data into the new coordinate space and then
    interpolating back into the original data.

    "Forward"
    By converting the coordinates, rather than by
    interpolating the data. As a result, the data will be totally unchanged by the conversion (if we do not
    apply a Jacobian correction), but the coordinates will no longer have equal spacing.

    This is only really useful for zero and one dimensional data because for two dimensional data, the coordinates
    must become two dimensional in order to fully specify every data point (this is true in generality, in 3D the
    coordinates must become 3D as well).

    The only exception to this is if the extra axes do not need to be k-space converted. As is the case where one
    of the dimensions is `cycle` or `delay`, for instance.

    :param arr:
    :return:
    """

    if isinstance(arr, xr.Dataset):
        warnings.warn('Remember to use a DataArray not a Dataset, attempting to extract spectrum')
        attrs = arr.attrs.copy()
        arr = normalize_to_spectrum(arr)
        arr.attrs.update(attrs)

    if forward:
        raise NotImplementedError('Forward conversion of datasets not supported. Coordinate conversion is. '
                                  'See `arpes.utilities.conversion.forward.convert_coordinates_to_kspace_forward`')

    has_eV = 'eV' in arr.dims

    # TODO be smarter about the resolution inference
    old_dims = list(deepcopy(arr.dims))
    remove_dims = ['eV', 'delay', 'cycle', 'temp', 'x', 'y']

    def unconvertible(dimension):
        if dimension in remove_dims:
            return True

        if 'volt' in dimension:
            return True

        return False

    removed = []

    for to_remove in arr.dims:
        if unconvertible(to_remove):
            removed.append(to_remove)
            old_dims.remove(to_remove)

    # This should always be true because otherwise we have no hope of carrying
    # through with the conversion
    if 'eV' in removed:
        removed.remove('eV') # This is put at the front as a standardization

    old_dims.sort()

    if len(old_dims) == 0:
        return arr # no need to convert, might be XPS or similar

    converted_dims = (['eV'] if has_eV else []) + {
        ('phi',): ['kp'],

        ('phi', 'theta'): ['kx', 'ky'],
        ('beta', 'phi'): ['kx', 'ky'],
        ('phi', 'psi'): ['kx', 'ky'],

        ('hv', 'phi'): ['kp', 'kz'],

        ('hv', 'phi', 'theta'): ['kx', 'ky', 'kz'],
        ('beta', 'hv', 'phi'): ['kx', 'ky', 'kz'],
        ('hv', 'phi', 'psi'): ['kx', 'ky', 'kz'],
    }.get(tuple(old_dims)) + removed

    convert_cls = {
        ('phi',): ConvertKp,

        ('beta', 'phi'): ConvertKxKy,
        ('phi', 'theta'): ConvertKxKy,
        ('phi', 'psi'): ConvertKxKy,
        #('chi', 'phi',): ConvertKxKy,

        ('hv', 'phi'): ConvertKpKz,
    }.get(tuple(old_dims))
    converter = convert_cls(arr, converted_dims)

    n_kspace_coordinates = len(set(converted_dims).intersection({'kp', 'kx', 'ky', 'kz'}))
    if n_kspace_coordinates > 1 and forward:
        raise AnalysisError('You cannot forward convert more than one momentum to k-space.')

    converted_coordinates = converter.get_coordinates(resolution)

    return convert_coordinates(
        arr, converted_coordinates, {
            'dims': converted_dims,
            'transforms': dict(zip(arr.dims, [converter.conversion_for(d) for d in arr.dims]))})[0]


def convert_coordinates(arr: xr.DataArray, target_coordinates, coordinate_transform, as_dataset=False):
    ordered_source_dimensions = arr.dims
    grid_interpolator = grid_interpolator_from_dataarray(
        arr.transpose(*ordered_source_dimensions), fill_value=float('nan'))

    # Skip the Jacobian correction for now
    # Convert the raw coordinate axes to a set of gridded points
    meshed_coordinates = np.meshgrid(*[target_coordinates[dim] for dim in coordinate_transform['dims']],
                                     indexing='ij')
    meshed_coordinates = [meshed_coord.ravel() for meshed_coord in meshed_coordinates]

    if 'eV' not in arr.dims:
        meshed_coordinates = [arr.S.lookup_offset_coord('eV')] + meshed_coordinates

    old_coord_names = [dim for dim in arr.dims if dim not in target_coordinates]
    old_coordinate_transforms = [coordinate_transform['transforms'][dim] for dim in arr.dims if dim not in target_coordinates]
    old_dimensions = [np.reshape(tr(*meshed_coordinates), [len(target_coordinates[d]) for d in coordinate_transform['dims']], order='C')
                      for tr in old_coordinate_transforms]

    ordered_transformations = [coordinate_transform['transforms'][dim] for dim in arr.dims]
    converted_volume = grid_interpolator(np.array([tr(*meshed_coordinates) for tr in ordered_transformations]).T)

    # Wrap it all up
    def acceptable_coordinate(c):
        # Currently we do this to filter out coordinates that are functions of the old angular dimensions,
        # we could forward convert these, but right now we do not
        try:
            if set(c.dims).issubset(coordinate_transform['dims']):
                return True
            else:
                return False
        except:
            return True

    target_coordinates = {k: v for k, v in target_coordinates.items() if acceptable_coordinate(v)}
    data = xr.DataArray(
        np.reshape(converted_volume, [len(target_coordinates[d]) for d in coordinate_transform['dims']], order='C'),
        target_coordinates,
        coordinate_transform['dims'],
        attrs=arr.attrs,
    )
    old_mapped_coords = [xr.DataArray(values, target_coordinates, coordinate_transform['dims'], attrs=arr.attrs) for values in
                         old_dimensions]
    if as_dataset:
        vars = {'data': data}
        vars.update(dict(zip(old_coord_names, old_mapped_coords)))
        return xr.Dataset(vars, attrs=arr.attrs)

    return data, old_mapped_coords
