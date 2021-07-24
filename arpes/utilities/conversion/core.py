"""Helper functions for coordinate transformations and user/analysis API.

All the functions here assume standard polar angles, as given in the
`data model documentation <https://arpes.readthedocs.io/spectra>`_.

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

Better facilities should be added for ToFs to do simultaneous (timing, angle) to (binding energy, k-space).
"""

from arpes.utilities.conversion.grids import (
    determine_axis_type,
    determine_momentum_axes_from_measurement_axes,
    is_dimension_unconvertible,
)
from .fast_interp import Interpolator

from arpes.trace import traceable
import collections
import warnings

import numpy as np
import scipy.interpolate

import xarray as xr
from arpes.provenance import provenance, update_provenance
from arpes.utilities import normalize_to_spectrum
from typing import Callable, Optional, Union

from .kx_ky_conversion import ConvertKxKy, ConvertKp
from .kz_conversion import ConvertKpKz

__all__ = ["convert_to_kspace", "slice_along_path"]


@traceable
def grid_interpolator_from_dataarray(
    arr: xr.DataArray,
    fill_value=0.0,
    method="linear",
    bounds_error=False,
    trace: Callable = None,
):
    """Translates an xarray.DataArray contents into a scipy.interpolate.RegularGridInterpolator.

    This is principally used for coordinate translations.
    """
    flip_axes = set()
    for d in arr.dims:
        c = arr.coords[d]
        if len(c) > 1 and c[1] - c[0] < 0:
            flip_axes.add(d)

    values = arr.values
    trace("Flipping axes")
    for dim in flip_axes:
        values = np.flip(values, arr.dims.index(dim))

    interp_points = [
        arr.coords[d].values[::-1] if d in flip_axes else arr.coords[d].values for d in arr.dims
    ]
    trace_size = [len(pts) for pts in interp_points]

    if method == "linear":
        trace(f"Using fast_interp.Interpolator: size {trace_size}")
        return Interpolator.from_arrays(interp_points, values)

    trace(f"Calling scipy.interpolate.RegularGridInterpolator: size {trace_size}")
    return scipy.interpolate.RegularGridInterpolator(
        points=interp_points,
        values=values,
        bounds_error=bounds_error,
        fill_value=fill_value,
        method=method,
    )


def slice_along_path(
    arr: xr.DataArray,
    interpolation_points=None,
    axis_name=None,
    resolution=None,
    shift_gamma=True,
    n_points: Optional[int] = None,
    extend_to_edge=False,
    **kwargs,
):
    """Gets a cut along a path specified by waypoints in an array.

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

    Args:
        arr: Source data
        interpolation_points: Path vertices
        axis_name: Label for the interpolated axis. Under special
            circumstances a reasonable name will be chosen,
        resolution: Requested resolution along the interpolated axis.
        shift_gamma: Controls whether the interpolated axis is shifted
            to a value of 0 at Gamma.
        n_points: The number of desired points along the output path. This will be inferred
            approximately based on resolution if not provided.
        extend_to_edge: Controls whether or not to scale the vector S -
            G for symmetry point S so that you interpolate
        **kwargs
    such as when the interpolation dimensions are kx and ky: in this case the interpolated dimension will be labeled kp.
    In mixed or ambiguous situations the axis will be labeled by the default value 'inter'.
    to the edge of the available data

    Returns:
        xr.DataArray containing the interpolated data.
    """
    if interpolation_points is None:
        raise ValueError("You must provide points specifying an interpolation path")

    def extract_symmetry_point(name):
        raw_point = arr.attrs["symmetry_points"][name]
        G = arr.attrs["symmetry_points"]["G"]

        if not extend_to_edge or name == "G":
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
        x
        if isinstance(x, collections.Iterable) and not isinstance(x, str)
        else extract_symmetry_point(x)
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
                    raise ValueError(
                        "Ambiguous interpolation waypoint broadcast at dimension {}".format(coord)
                    )
                else:
                    point[coord] = list(values)[0]

    if axis_name is None:
        try:
            axis_name = determine_axis_type(seen_coordinates.keys())
        except KeyError:
            axis_name = "inter"

        if axis_name == "angle" or axis_name == "inter":
            warnings.warn(
                "Interpolating along axes with different dimensions "
                "will not include Jacobian correction factor."
            )

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

    gamma_offset = 0  # offset the gamma point to a k coordinate of 0 if possible
    if "G" in interpolation_points and shift_gamma:
        gamma_offset = sum(segment_lengths[0 : interpolation_points.index("G")])

    if resolution is None:
        if n_points is None:
            resolution = np.min([required_sampling_density(*segment) for segment in path_segments])
        else:
            path_length / n_points

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
                dest_coordinate[mask] = (
                    dim_start * (1 - normalized[mask]) + dim_end * normalized[mask]
                )
                start = end

            return dest_coordinate

        return interpolated_coordinate_to_raw

    converted_coordinates = {d: arr.coords[d].values for d in free_coordinates}

    if n_points is None:
        n_points = int(path_length / resolution)

    # Adjust this coordinate under special circumstances
    converted_coordinates[axis_name] = (
        np.linspace(0, path_length, int(path_length / resolution)) - gamma_offset
    )

    converted_ds = convert_coordinates(
        arr,
        converted_coordinates,
        {
            "dims": converted_dims,
            "transforms": dict(zip(arr.dims, [converter_for_coordinate_name(d) for d in arr.dims])),
        },
        as_dataset=True,
    )

    if axis_name in arr.dims and len(parsed_interpolation_points) == 2:
        if parsed_interpolation_points[1][axis_name] < parsed_interpolation_points[0][axis_name]:
            # swap the sign on this axis as a convenience to the caller
            converted_ds.coords[axis_name].data = -converted_ds.coords[axis_name].data

    if "id" in converted_ds.attrs:
        del converted_ds.attrs["id"]
        provenance(
            converted_ds,
            arr,
            {
                "what": "Slice along path",
                "by": "slice_along_path",
                "parsed_interpolation_points": parsed_interpolation_points,
                "interpolation_points": interpolation_points,
            },
        )

    return converted_ds


@update_provenance("Automatically k-space converted")
@traceable
def convert_to_kspace(
    arr: xr.DataArray,
    bounds=None,
    resolution=None,
    calibration=None,
    coords=None,
    allow_chunks: bool = False,
    trace: Callable = None,
    **kwargs,
):
    """Converts volumetric the data to momentum space ("backwards"). Typically what you want.

    Works in general by regridding the data into the new coordinate space and then
    interpolating back into the original data.

    For forward conversion, see sibling methods. Forward conversion works by
    converting the coordinates, rather than by interpolating
    the data. As a result, the data will be totally unchanged by the conversion
    (if we do not apply a Jacobian correction), but the coordinates will no
    longer have equal spacing.

    This is only really useful for zero and one dimensional data because for two dimensional data,
    the coordinates must become two dimensional in order to fully specify every data point
    (this is true in generality, in 3D the coordinates must become 3D as well).

    The only exception to this is if the extra axes do not need to be k-space converted. As is the
    case where one of the dimensions is `cycle` or `delay`, for instance.

    You can request a particular resolution for the new data with the `resolution=` parameter,
    or a specific set of bounds with the `bounds=`

    Examples:
        Convert a 2D cut with automatically inferred range and resolution.

        >>> convert_to_kspace(arpes.io.load_example_data())  # doctest: +SKIP
        xr.DataArray(...)

        Convert a 3D map with a specified momentum window

        >>> convert_to_kspace(  # doctest: +SKIP
                fermi_surface_map,
                kx=np.linspace(-1, 1, 200),
                ky=np.linspace(-1, 1, 350),
            )
        xr.DataArray(...)

    Args:
        arr (xr.DataArray): [description]
        #bounds ([type], optional): [description]. Defaults to None.
        resolution ([type], optional): [description]. Defaults to None.
        calibration ([type], optional): [description]. Defaults to None.
        coords ([type], optional): [description]. Defaults to None.
        allow_chunks (bool, optional): [description]. Defaults to False.
        trace (Callable, optional): Controls whether to use execution tracing. Defaults to None.
          Pass `True` to enable.

    Raises:
        NotImplementedError: [description]
        AnalysisError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if coords is None:
        coords = {}

    coords.update(kwargs)

    trace("Normalizing to spectrum")
    if isinstance(arr, xr.Dataset):
        warnings.warn(
            "Remember to use a DataArray not a Dataset, attempting to extract spectrum and copy attributes."
        )
        attrs = arr.attrs.copy()
        arr = normalize_to_spectrum(arr)
        arr.attrs.update(attrs)

    has_eV = "eV" in arr.dims

    # Chunking logic
    if allow_chunks and has_eV and len(arr.eV) > 50:
        DESIRED_CHUNK_SIZE = 1000 * 1000 * 20
        n_chunks = np.prod(arr.shape) // DESIRED_CHUNK_SIZE
        if n_chunks > 100:
            warnings.warn("Input array is very large. Please consider resampling.")

        chunk_thickness = max(len(arr.eV) // n_chunks, 1)

        trace(f"Chunking along energy: {n_chunks}, thickness {chunk_thickness}")

        finished = []
        low_idx = 0
        high_idx = chunk_thickness

        while low_idx < len(arr.eV):
            chunk = arr.isel(eV=slice(low_idx, high_idx))

            if len(chunk.eV) == 1:
                chunk = chunk.squeeze("eV")

            kchunk = convert_to_kspace(
                chunk,
                bounds=bounds,
                resolution=resolution,
                calibration=calibration,
                coords=coords,
                allow_chunks=False,
                trace=trace,
                **kwargs,
            )

            if "eV" not in kchunk.dims:
                kchunk = kchunk.expand_dims("eV")

            finished.append(kchunk)

            low_idx = high_idx
            high_idx = min(len(arr.eV), high_idx + chunk_thickness)

        return xr.concat(finished, dim="eV")

    # Chunking is finished here

    # TODO be smarter about the resolution inference
    trace("Determining dimensions and resolution")
    removed = [d for d in arr.dims if is_dimension_unconvertible(d)]
    old_dims = [d for d in arr.dims if not is_dimension_unconvertible(d)]

    # Energy gets put at the front as a standardization
    if "eV" in removed:
        removed.remove("eV")

    old_dims.sort()

    trace("Replacing dummy coordinates with index-like ones.")
    # temporarily reassign coordinates for dimensions we will not
    # convert to "index-like" dimensions
    restore_index_like_coordinates = {r: arr.coords[r].values for r in removed}
    new_index_like_coordinates = {r: np.arange(len(arr.coords[r].values)) for r in removed}
    arr = arr.assign_coords(**new_index_like_coordinates)

    if not old_dims:
        return arr  # no need to convert, might be XPS or similar

    converted_dims = (
        (["eV"] if has_eV else [])
        + determine_momentum_axes_from_measurement_axes(old_dims)
        + removed
    )

    convert_cls = {
        ("phi",): ConvertKp,
        ("beta", "phi"): ConvertKxKy,
        ("phi", "theta"): ConvertKxKy,
        ("phi", "psi"): ConvertKxKy,
        # ('chi', 'phi',): ConvertKxKy,
        ("hv", "phi"): ConvertKpKz,
    }.get(tuple(old_dims))
    converter = convert_cls(arr, converted_dims, calibration=calibration)

    trace("Converting coordinates")
    converted_coordinates = converter.get_coordinates(resolution=resolution, bounds=bounds)

    if not set(coords.keys()).issubset(converted_coordinates.keys()):
        extra = set(coords.keys()).difference(converted_coordinates.keys())
        raise ValueError("Unexpected passed coordinates: {}".format(extra))

    converted_coordinates.update(coords)

    trace("Calling convert_coordinates")
    result = convert_coordinates(
        arr,
        converted_coordinates,
        {
            "dims": converted_dims,
            "transforms": dict(zip(arr.dims, [converter.conversion_for(d) for d in arr.dims])),
        },
        trace=trace,
    )
    trace("Reassigning index-like coordinates.")
    result = result.assign_coords(**restore_index_like_coordinates)
    trace("Finished.")
    return result


@traceable
def convert_coordinates(
    arr: xr.DataArray,
    target_coordinates,
    coordinate_transform,
    as_dataset=False,
    trace: Callable = None,
):
    ordered_source_dimensions = arr.dims
    trace("Instantiating grid interpolator.")
    grid_interpolator = grid_interpolator_from_dataarray(
        arr.transpose(*ordered_source_dimensions),
        fill_value=float("nan"),
        trace=trace,
    )
    trace("Finished instantiating grid interpolator.")

    # Skip the Jacobian correction for now
    # Convert the raw coordinate axes to a set of gridded points
    trace(f"Calling meshgrid: {[len(target_coordinates[d]) for d in coordinate_transform['dims']]}")
    meshed_coordinates = np.meshgrid(
        *[target_coordinates[dim] for dim in coordinate_transform["dims"]], indexing="ij"
    )
    trace("Raveling coordinates")
    meshed_coordinates = [meshed_coord.ravel() for meshed_coord in meshed_coordinates]

    if "eV" not in arr.dims:
        try:
            meshed_coordinates = [arr.S.lookup_offset_coord("eV")] + meshed_coordinates
        except ValueError:
            pass

    old_coord_names = [dim for dim in arr.dims if dim not in target_coordinates]
    old_coordinate_transforms = [
        coordinate_transform["transforms"][dim] for dim in arr.dims if dim not in target_coordinates
    ]

    trace(f"Calling coordinate transforms")
    output_shape = [len(target_coordinates[d]) for d in coordinate_transform["dims"]]

    def compute_coordinate(transform):
        return np.reshape(
            transform(*meshed_coordinates),
            output_shape,
            order="C",
        )

    old_dimensions = []
    for tr in old_coordinate_transforms:
        trace(f"Running transform {tr}")
        old_dimensions.append(compute_coordinate(tr))

    trace(f"Done running transforms.")

    ordered_transformations = [coordinate_transform["transforms"][dim] for dim in arr.dims]
    trace("Calling grid interpolator")

    trace("Pulling back coordinates")
    transformed_coordinates = []
    for tr in ordered_transformations:
        trace(f"Running transform {tr}")
        transformed_coordinates.append(tr(*meshed_coordinates))

    if not isinstance(grid_interpolator, Interpolator):
        transformed_coordinates = np.array(transformed_coordinates).T

    trace("Calling grid interpolator")
    converted_volume = grid_interpolator(transformed_coordinates)

    # Wrap it all up
    def acceptable_coordinate(c: Union[np.ndarray, xr.DataArray]) -> bool:
        # Currently we do this to filter out coordinates that are functions of the old angular dimensions,
        # we could forward convert these, but right now we do not
        try:
            if set(c.dims).issubset(coordinate_transform["dims"]):
                return True
            else:
                return False
        except:
            return True

    trace("Bundling into DataArray")
    target_coordinates = {k: v for k, v in target_coordinates.items() if acceptable_coordinate(v)}
    data = xr.DataArray(
        np.reshape(
            converted_volume,
            [len(target_coordinates[d]) for d in coordinate_transform["dims"]],
            order="C",
        ),
        target_coordinates,
        coordinate_transform["dims"],
        attrs=arr.attrs,
    )
    old_mapped_coords = [
        xr.DataArray(values, target_coordinates, coordinate_transform["dims"], attrs=arr.attrs)
        for values in old_dimensions
    ]
    if as_dataset:
        vars = {"data": data}
        vars.update(dict(zip(old_coord_names, old_mapped_coords)))
        return xr.Dataset(vars, attrs=arr.attrs)

    trace("Finished")
    return data
