"""Contains routines for converting directly from angle to momentum.

This cannot be done easily for volumetric data because otherwise we will 
not end up with an even grid. As a result, we typically use utilities here
to look at the forward projection of a single point or collection of 
points/coordinates under the angle -> momentum transform.

Additionally, we have exact inverses for the volumetric transforms which are 
useful for aligning cuts which use those transforms. 
See `convert_coordinate_forward`.
"""
from arpes.utilities.conversion.core import convert_to_kspace
from typing import Callable, Dict
from arpes.trace import traceable
import numpy as np
import warnings
import xarray as xr

from arpes.utilities import normalize_to_spectrum
from arpes.provenance import update_provenance
from arpes.analysis.filters import gaussian_filter_arr
from arpes.utilities.conversion.bounds_calculations import (
    euler_to_kx,
    euler_to_ky,
    euler_to_kz,
    full_angles_to_k,
)
from arpes.typing import DataType

__all__ = (
    "convert_coordinates_to_kspace_forward",
    "convert_coordinates",
    "convert_coordinate_forward",
    "convert_through_angular_pair",
    "convert_through_angular_point",
)


@traceable
def convert_coordinate_forward(
    data: DataType, coords: Dict[str, float], trace: Callable = None, **k_coords
):
    """This function is the inverse/forward transform for the small angle volumetric k-conversion code.

    This differs from the other forward transforms here which are exact, up to correct assignment
    of offset constants.

    This makes this routine very practical for determining the location of cuts to be taken around a
    point or direction of interest in k-space. If you use the exact methods to determine the
    location of interest in k-space then in general there will be some misalignment because
    the small angle volumetric transform is not the inverse of the exact forward transforms.

    The way that we accomplish this is that the data is copied and a "test charge" is placed in the
    data which distinguished the location of interest in angle-space. The data is converted with
    the volumetric interpolation methods, and the location of the "test charge" is determined in
    k-space. With the approximate location in k determined, this process is repeated once more with a
    finer k-grid to determine more precisely the forward transform location.

    A nice property of this approach is that it is automatic because it determines the result
    numerically using the volumetric transform code. Any changes to the volumetric code will
    automatically reflect here. However, it comes with a few downsides:

    #. The "test charge" is placed in a cell in the original data. This means that the resolution is
       limited by the resolution of the dataset in angle-space. This could be circumvented by
       regridding the data to have a higher resolution.
    #. The procedure can only be performed for a single point at a time.
    #. The procedure is relatively expensive.

    Another approach would be to write down the exact small angle approximated transforms.

    Args:
        data: The data defining the coordinate offsets and experiment geometry.
        coords: The coordinates of a point in angle-space to be converted.
        trace: Used for performance tracing and debugging.

    Returns:
        The location of the desired coordinate in momentum.
    """
    data = normalize_to_spectrum(data)

    if "eV" in coords:
        coords = dict(coords)
        energy_coord = coords.pop("eV")
        data = data.sel(eV=energy_coord, method="nearest")
    elif "eV" in data.dims:
        warnings.warn(
            """
            You didn't specify an energy coordinate for the high symmetry point but the 
            dataset you provided has an energy dimension. This will likely be very 
            slow. Where possible, provide an energy coordinate
            """
        )

    if not k_coords:
        k_coords = {
            "kx": np.linspace(-4, 4, 300),
            "ky": np.linspace(-4, 4, 300),
        }

    # Copying after taking a constant energy plane is much much cheaper
    trace("Copying")
    data = data.copy(deep=True)

    data.loc[data.G.round_coordinates(coords)] = data.values.max() * 100000
    trace("Filtering")
    data = gaussian_filter_arr(data, default_size=3)

    trace("Converting once")
    kdata = convert_to_kspace(data, **k_coords, trace=trace)

    trace("argmax")
    near_target = kdata.G.argmax_coords()

    trace("Converting twice")
    kdata_close = convert_to_kspace(
        data,
        trace=trace,
        **{k: np.linspace(v - 0.08, v + 0.08, 100) for k, v in near_target.items()},
    )

    # inconsistently, the energy coordinate is sometimes returned here
    # so we remove it just in case
    trace("argmax")
    coords = kdata_close.G.argmax_coords()
    if "eV" in coords:
        del coords["eV"]
    return coords


@traceable
def convert_through_angular_pair(
    data: DataType,
    first_point: Dict[str, float],
    second_point: Dict[str, float],
    cut_specification: Dict[str, np.ndarray],
    transverse_specification: Dict[str, np.ndarray],
    relative_coords: bool = True,
    trace: Callable = None,
    **k_coords,
):
    """Converts the lower dimensional ARPES cut passing through `first_point` and `second_point`.

    This is a sibling method to `convert_through_angular_point`. A point and a `chi` angle
    fix a plane in momentum, as do two points. This method implements the two points
    equivalent whereas `convert_through_angular_point` fixes the latter.

    A difference between this function and `convert_through_angular_point` is that
    the momentum range requested by `cut_specification` is considered a margin outside
    the interval defined by `first_point` and `second_point`. I.e.
    np.linspace(0, 0, 400) would produce a 400 point interpolation between
    `first_point` and `second_point` whereas passing `np.linspace(-1, 0, 400)` would provide
    a left margin past `first_point` of 1 inverse angstrom.

    This endpoint relative behavior can be disabled with the `relative_coords` flag.

    Args:
        data: The angle space data to be converted.
        first_point: The angle space coordinates of the first point the cut should
          pass through.
        second_point: The angle space coordinates of the second point the cut should
          pass through. This point will have larger momentum coordinate in the output
          data than `first_point`, i.e. it will appear on the "right side" of the data
          if the data is plotted as a cut.
        cut_specification: A dictionary specifying the momentum varying axes
          on the output data. Interpreted as defining boundaries relative to
          the endpoints.
        transverse_specification: A dictionary specifying the transverse (summed)
          axis on the momentum converted data.
        relative_coords: Whether to give `cut_specification` relative to the momentum
          converted location specified in `coords`
        trace: Flag controlling execution tracing
        k_coords: Passed as hints through to `convert_coordinate_forward`.

    Returns:
        The momentum cut passing first through `first_point` and then through `second_point`.
    """
    k_first_point = convert_coordinate_forward(data, first_point, trace=trace, **k_coords)
    k_second_point = convert_coordinate_forward(data, second_point, trace=trace, **k_coords)

    k_dims = set(k_first_point.keys())
    if k_dims != {"kx", "ky"}:
        raise NotImplementedError(f"Two point {k_dims} momentum conversion is not supported yet.")

    assert k_dims == set(cut_specification.keys()).union(transverse_specification.keys())
    assert (
        "ky" in transverse_specification.keys()
    )  # You must use ky as the transverse coordinate for now
    assert len(cut_specification) == 1

    offset_ang = np.arctan2(
        k_second_point["ky"] - k_first_point["ky"],
        k_second_point["kx"] - k_first_point["kx"],
    )

    with data.S.with_rotation_offset(-offset_ang):
        k_first_point = convert_coordinate_forward(data, first_point, trace=trace, **k_coords)
        k_second_point = convert_coordinate_forward(data, second_point, trace=trace, **k_coords)

        # adjust output coordinate ranges
        transverse_specification = {
            k: v + k_first_point[k] for k, v in transverse_specification.items()
        }

        # here, we assume we were passed an array for simplicities sake
        # we could also allow a slice in the future
        parallel_axis = list(cut_specification.values())[0]
        if relative_coords:
            delta = parallel_axis[1] - parallel_axis[0]
            left_margin, right_margin = parallel_axis[0], parallel_axis[-1] + delta

            left_point = k_first_point["kx"] + left_margin
            right_point = k_second_point["kx"] + right_margin
            parallel_axis = np.linspace(left_point, right_point, len(parallel_axis))

        # perform the conversion
        return convert_to_kspace(data, **transverse_specification, kx=parallel_axis).mean(
            list(transverse_specification.keys())
        )


@traceable
def convert_through_angular_point(
    data: DataType,
    coords: Dict[str, float],
    cut_specification: Dict[str, np.ndarray],
    transverse_specification: Dict[str, np.ndarray],
    relative_coords: bool = True,
    trace: Callable = None,
    **k_coords,
) -> xr.DataArray:
    """Converts the lower dimensional ARPES cut passing through given angular `coords`.

    The fixed momentum axis is given by `cut_specification` in coordinates relative to
    the momentum converted copy of `coords`. Absolute coordinates can be enabled
    with the `relative_coords` flag.


    Args:
        data: The angle space data to be converted.
        coords: The angle space coordinates of the point the cut should pass through.
        cut_specification: A dictionary specifying the momentum varying axes
          on the output data.
        transverse_specification: A dictionary specifying the transverse (summed)
          axis on the momentum converted data.
        relative_coords: Whether to give `cut_specification` relative to the momentum
          converted location specified in `coords`
        trace: Flag controlling execution tracing
        k_coords: Passed as hints through to `convert_coordinate_forward`.

    Returns:
        A momentum cut passing through the point `coords`.
    """
    k_coords = convert_coordinate_forward(data, coords, trace=trace, **k_coords)
    all_momentum_dims = set(k_coords.keys())
    assert all_momentum_dims == set(cut_specification.keys()).union(transverse_specification.keys())

    # adjust output coordinate ranges
    transverse_specification = {k: v + k_coords[k] for k, v in transverse_specification.items()}
    if relative_coords:
        cut_specification = {k: v + k_coords[k] for k, v in cut_specification.items()}

    # perform the conversion
    return convert_to_kspace(data, **transverse_specification, **cut_specification).mean(
        list(transverse_specification.keys())
    )


@update_provenance("Forward convert coordinates")
def convert_coordinates(arr: DataType, collapse_parallel=False, **kwargs):
    """Converts coordinates forward in momentum."""

    def unwrap_coord(c):
        try:
            return c.values
        except (TypeError, AttributeError):
            try:
                return c.item()
            except (TypeError, AttributeError):
                return c

    coord_names = ["phi", "psi", "alpha", "theta", "beta", "chi", "hv"]
    raw_coords = {k: unwrap_coord(arr.S.lookup_offset_coord(k)) for k in (coord_names + ["eV"])}
    raw_angles = {k: v for k, v in raw_coords.items() if k not in {"eV", "hv"}}

    parallel_collapsible = (
        len([k for k in raw_angles.keys() if isinstance(raw_angles[k], np.ndarray)]) > 1
    )

    sort_by = ["eV", "hv", "phi", "psi", "alpha", "theta", "beta", "chi"]
    old_dims = sorted(
        [k for k in arr.dims if k in (coord_names + ["eV"])], key=lambda item: sort_by.index(item)
    )

    will_collapse = parallel_collapsible and collapse_parallel

    def expand_to(cname, c):
        if not isinstance(c, np.ndarray):
            return c

        index_list = [np.newaxis] * len(old_dims)
        index_list[old_dims.index(cname)] = slice(None, None)
        return c[tuple(index_list)]

    # build the full kinetic energy array over relevant dimensions
    kinetic_energy = (
        expand_to("eV", raw_coords["eV"]) + expand_to("hv", raw_coords["hv"]) - arr.S.work_function
    )

    kx, ky, kz = full_angles_to_k(
        kinetic_energy,
        inner_potential=arr.S.inner_potential,
        **{k: expand_to(k, v) for k, v in raw_angles.items()},
    )

    if will_collapse:
        if np.sum(kx ** 2) > np.sum(ky ** 2):
            sign = kx / np.sqrt(kx ** 2 + 1e-8)
        else:
            sign = ky / np.sqrt(ky ** 2 + 1e-8)

        kp = sign * np.sqrt(kx ** 2 + ky ** 2)
        data_vars = {"kp": (old_dims, np.squeeze(kp)), "kz": (old_dims, np.squeeze(kz))}
    else:
        data_vars = {
            "kx": (old_dims, np.squeeze(kx)),
            "ky": (old_dims, np.squeeze(ky)),
            "kz": (old_dims, np.squeeze(kx)),
        }

    return xr.Dataset(data_vars, coords=arr.indexes)


@update_provenance("Forward convert coordinates to momentum")
def convert_coordinates_to_kspace_forward(arr: DataType, **kwargs):
    """Forward converts all the individual coordinates of the data array."""
    arr = arr.copy(deep=True)

    skip = {"eV", "cycle", "delay", "T"}
    keep = {
        "eV",
    }

    all = {k: v for k, v in arr.indexes.items() if k not in skip}
    kept = {k: v for k, v in arr.indexes.items() if k in keep}

    old_dims = list(all.keys())
    old_dims.sort()

    if not old_dims:
        return None

    dest_coords = {
        ("phi",): ["kp", "kz"],
        ("theta",): ["kp", "kz"],
        ("beta",): ["kp", "kz"],
        ("phi", "theta"): ["kx", "ky", "kz"],
        ("beta", "phi"): ["kx", "ky", "kz"],
        ("hv", "phi"): ["kx", "ky", "kz"],
        ("hv",): ["kp", "kz"],
        ("beta", "hv", "phi"): ["kx", "ky", "kz"],
        ("hv", "phi", "theta"): ["kx", "ky", "kz"],
        ("hv", "phi", "psi"): ["kx", "ky", "kz"],
        ("chi", "hv", "phi"): ["kx", "ky", "kz"],
    }.get(tuple(old_dims))

    full_old_dims = old_dims + list(kept.keys())
    projection_vectors = np.ndarray(
        shape=tuple(len(arr.coords[d]) for d in full_old_dims), dtype=object
    )

    # these are a little special, depending on the scan type we might not have a phi coordinate
    # that aspect of this is broken for now, but we need not worry
    def broadcast_by_dim_location(data, target_shape, dim_location=None):
        if isinstance(data, xr.DataArray):
            if not data.dims:
                data = data.item()

        if isinstance(
            data,
            (
                int,
                float,
            ),
        ):
            return np.ones(target_shape) * data

        # else we are dealing with an actual array
        the_slice = [None] * len(target_shape)
        the_slice[dim_location] = slice(None, None, None)

        return np.asarray(data)[the_slice]

    raw_coords = {
        "phi": arr.coords["phi"].values - arr.S.phi_offset,
        "beta": (0 if arr.coords["beta"] is None else arr.coords["beta"].values)
        - arr.S.beta_offset,
        "theta": (0 if arr.coords["theta"] is None else arr.coords["theta"].values)
        - arr.S.theta_offset,
        "hv": arr.coords["hv"],
    }

    raw_coords = {
        k: broadcast_by_dim_location(
            v, projection_vectors.shape, full_old_dims.index(k) if k in full_old_dims else None
        )
        for k, v in raw_coords.items()
    }

    # fill in the vectors
    binding_energy = broadcast_by_dim_location(
        arr.coords["eV"] - arr.S.work_function,
        projection_vectors.shape,
        full_old_dims.index("eV") if "eV" in full_old_dims else None,
    )
    photon_energy = broadcast_by_dim_location(
        arr.coords["hv"],
        projection_vectors.shape,
        full_old_dims.index("hv") if "hv" in full_old_dims else None,
    )
    kinetic_energy = binding_energy + photon_energy

    inner_potential = arr.S.inner_potential

    # some notes on angle conversion:
    # BL4 conventions
    # angle conventions are standard:
    # phi = analyzer acceptance
    # polar = perpendicular scan angle
    # theta = parallel to analyzer slit rotation angle

    # [ 1  0          0          ]   [  cos(polar) 0 sin(polar) ]   [ 0          ]
    # [ 0  cos(theta) sin(theta) ] * [  0          1 0          ] * [ k sin(phi) ]
    # [ 0 -sin(theta) cos(theta) ]   [ -sin(polar) 0 cos(polar) ]   [ k cos(phi) ]
    #
    # =
    #
    # [ 1  0          0          ]     [ sin(polar) * cos(phi) ]
    # [ 0  cos(theta) sin(theta) ] * k [ sin(phi) ]
    # [ 0 -sin(theta) cos(theta) ]     [ cos(polar) * cos(phi) ]
    #
    # =
    #
    # k ( sin(polar) * cos(phi),
    #     cos(theta)*sin(phi) + cos(polar) * cos(phi) * sin(theta),
    #     -sin(theta) * sin(phi) + cos(theta) * cos(polar) * cos(phi),
    #   )
    #
    # main chamber conventions, with no analyzer rotation (referred to as alpha angle in the Igor code
    # angle conventions are standard:
    # phi = analyzer acceptance
    # polar = perpendicular scan angle
    # theta = parallel to analyzer slit rotation angle

    # [ 1 0 0                    ]     [ sin(phi + theta) ]
    # [ 0 cos(polar) sin(polar)  ] * k [ 0                ]
    # [ 0 -sin(polar) cos(polar) ]     [ cos(phi + theta) ]
    #
    # =
    #
    # k (sin(phi + theta), cos(phi + theta) * sin(polar), cos(phi + theta) cos(polar), )
    #

    # for now we are setting the theta angle to zero, this only has an effect for vertical slit analyzers,
    # and then only when the tilt angle is very large

    # TODO check me
    raw_translated = {
        "kx": euler_to_kx(
            kinetic_energy,
            raw_coords["phi"],
            raw_coords["beta"],
            theta=0,
            slit_is_vertical=arr.S.is_slit_vertical,
        ),
        "ky": euler_to_ky(
            kinetic_energy,
            raw_coords["phi"],
            raw_coords["beta"],
            theta=0,
            slit_is_vertical=arr.S.is_slit_vertical,
        ),
        "kz": euler_to_kz(
            kinetic_energy,
            raw_coords["phi"],
            raw_coords["beta"],
            theta=0,
            slit_is_vertical=arr.S.is_slit_vertical,
            inner_potential=inner_potential,
        ),
    }

    if "kp" in dest_coords:
        if np.sum(raw_translated["kx"] ** 2) > np.sum(raw_translated["ky"] ** 2):
            sign = raw_translated["kx"] / np.sqrt(raw_translated["kx"] ** 2 + 1e-8)
        else:
            sign = raw_translated["ky"] / np.sqrt(raw_translated["ky"] ** 2 + 1e-8)

        raw_translated["kp"] = np.sqrt(raw_translated["kx"] ** 2 + raw_translated["ky"] ** 2) * sign

    data_vars = {}
    for dest_coord in dest_coords:
        data_vars[dest_coord] = (full_old_dims, np.squeeze(raw_translated[dest_coord]))

    return xr.Dataset(data_vars, coords=arr.indexes)
