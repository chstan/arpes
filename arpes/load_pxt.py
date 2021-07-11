"""Implements Igor <-> xarray interop, notably loading Igor waves and packed experiment files."""

import re
import typing
import warnings

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import xarray as xr

from arpes.utilities.string import safe_decode
from arpes.typing import DataType

Wave = Any  # really, igor.Wave but we do not assume installation

__all__ = (
    "read_single_pxt",
    "read_separated_pxt",
    "read_experiment",
    "find_ses_files_associated",
)

binary_header_bytes = 10

igor_wave_header_dtype = np.dtype(
    [
        ("next", ">u8"),
        ("creation_date", ">u8"),
        ("mod_date", ">u8"),
        ("type", ">H"),
        ("d_lock", ">H"),
        ("wave_header_pad1", "B", (62,)),  # unused
        ("n_points", ">u8"),
        ("wave_header_version", ">u4"),
        ("spacer", ">u8"),
        ("wave_name", "S", 32),
        ("wave_header_pad2", "B", (8,)),
        ("dim_sizes", ">u4", (4,)),
        ("dim_scales", ">d", (4,)),
        ("dim_offsets", ">d", (4,)),
        ("data_units", "b", (4,)),
        ("dim_units", ("S", 4), (4,)),
        ("final_spacer", "b", 2 + 8 * 18),
    ]
)


def read_igor_binary_wave(raw_bytes: bytes) -> xr.DataArray:
    """Reads an igor wave from raw binary data using the documented Igor binary format.

    Some weirdness here with the byte ordering and the target datatype. Additionally,
    newer Igor wave formats are very poorly documented. Data loading might not be perfectly
    correct as a result. For most purposes, you can load from .pxp files anyway.

    Roughly, we first read a header from the bytestream. This header is used to determine the
    dtype and size of the array which remains to be read from the tail of the bytestream.

    The header is defined by `igor_wave_header_dtype`.

    Args:
        raw_bytes: The bytes/buffer to be read.

    Returns:
        The array read from the bytestream as an `xr.DataArray`.
    """
    header = np.fromstring(
        raw_bytes[: igor_wave_header_dtype.itemsize], dtype=igor_wave_header_dtype
    ).item()

    point_size = 8
    n_points = header[6]
    _ = header[9]  # wave_name
    dim_sizes, dim_scales, dim_offsets = header[11], header[12], header[13]
    _ = header[14]  # data_units
    dim_units = header[15]

    # approximate offsets
    # >i8 -1
    # <i8 0
    # >u8 -1
    # <u8 0
    offset = -1

    wave_data = np.fromstring(
        raw_bytes[
            igor_wave_header_dtype.itemsize
            + offset : igor_wave_header_dtype.itemsize
            + n_points * point_size
            + offset
        ],
        dtype=">u8",
    )

    wave_data = wave_data / np.max(wave_data)
    wave_data[wave_data == 1] = 0  # some weird bit shifting going on?

    names_from_units = {
        "eV": "eV",
        "deg": "phi",
    }

    dim_sizes = [d for d in dim_sizes if d]
    n_dims = len(dim_sizes)

    # please pylint forgive me
    dims = [
        (
            names_from_units.get(
                safe_decode(dim_units[i], prefer="ascii"), safe_decode(dim_units[i], prefer="ascii")
            ),
            np.linspace(
                dim_offsets[i], dim_offsets[i] + (dim_sizes[i] - 1) * dim_scales[i], dim_sizes[i]
            ),
        )
        for i in range(n_dims)
    ]
    coords = dict(dims)

    return xr.DataArray(
        wave_data.reshape(dim_sizes[::-1]),
        coords=coords,
        dims=[d[0] for d in dims][::-1],
    )


def read_header(header_bytes: bytes):
    header_as_string = safe_decode(header_bytes)

    lines = [x for x in header_as_string.replace("\r", "\n").split("\n") if x]
    lines = [x for x in lines if "=" in x]

    header = {}
    for line in lines:
        fragments = line.split("=")
        first, rest = fragments[0], "=".join(fragments[1:])

        try:
            rest = int(rest)
        except ValueError:
            try:
                rest = float(rest)
            except ValueError:
                pass

        header[first.lower().replace(" ", "_")] = rest

    from arpes.utilities import rename_keys

    return rename_keys(
        header,
        {
            "sample_x": "x",
            "sample_y_(vert)": "y",
            "sample_y": "y",
            "sample_z": "z",
            "bl_energy": "hv",
        },
    )


def wave_to_xarray(wave: Wave) -> xr.DataArray:
    """Converts a wave to an `xr.DataArray`.

    Units, if present on the wave, are used to furnish the dimension names.
    If dimension names are not present, placeholder names ("X", "Y", "Z", "W", as in Igor)
    are used for each unitless dimension.

    Args:
        wave: The input wave, an `igor.Wave` instance.

    Returns:
        The converted `xr.DataArray` instance.
    """
    # only need four because Igor only supports four dimensions!
    extra_names = iter(["W", "X", "Y", "Z"])
    n_dims = len([a for a in wave.axis if len(a)])

    def get_axis_name(index: int) -> str:
        unit = wave.axis_units[index]
        if unit:
            return {
                "eV": "eV",
                "deg": "phi",
                "Pwr Supply V": "volts",
                "K2200 V": "volts",
            }.get(unit, unit)

        return next(extra_names)

    axis_names = [get_axis_name(i) for i in range(n_dims)]
    coords = dict(zip(axis_names, wave.axis))

    return xr.DataArray(
        wave.data,
        coords=coords,
        dims=axis_names,
        attrs=read_header(wave.notes),
    )


def read_experiment(reference_path: typing.Union[Path, str], **kwargs) -> xr.Dataset:
    """Reads an entire Igor experiment to a set of waves, as an `xr.Dataset`.

    Looks for waves inside the experiment and collates them into an xr.Dataset using their
    Igor names as the var names for the `xr.Dataset`.

    Args:
        reference_path: The path to the experiment to be loaded.

    Returns:
        The loaded dataset with only waves retained..
    """
    import igor.igorpy as igor

    if isinstance(reference_path, Path):
        reference_path = str(reference_path.absolute())

    return igor.load(reference_path, **kwargs)


def read_single_ibw(reference_path: typing.Union[Path, str]) -> Wave:
    """Uses igor.igorpy to load an .ibw file."""
    import igor.igorpy as igor

    if isinstance(reference_path, Path):
        reference_path = str(reference_path.absolute())
    return igor.load(reference_path)


def read_single_pxt(reference_path: typing.Union[Path, str], byte_order=None) -> xr.DataArray:
    """Uses igor.igorpy to load a single .PXT or .PXP file."""
    import igor.igorpy as igor

    if isinstance(reference_path, Path):
        reference_path = str(reference_path.absolute())

    loaded = None
    if byte_order is None:
        for try_byte_order in [">", "=", "<"]:
            try:
                loaded = igor.load(reference_path, initial_byte_order=try_byte_order)
                break
            except Exception:  # pylint: disable=broad-except
                # bad byte ordering probably
                pass
    else:
        loaded = igor.load(reference_path, initial_byte_order=byte_order)

    children = [c for c in loaded.children if isinstance(c, igor.Wave)]

    if len(children) > 1:
        warnings.warn("Igor PXT file contained {} waves. Ignoring all but first.", len(children))

    return wave_to_xarray(children[0])


def find_ses_files_associated(reference_path: Path, separator: str = "S") -> List[Path]:
    """Finds all .pxt files created by in an SES scan, posfixed like "_S[0-9][0-9][0-9].pxt".

    SES Software creates a series of .pxt files. To load one of these scans we need to collect
    all the relevant files before loading them together. Typically, they are all sequenced
    with _S[0-9][0-9][0-9].pxt.

    This convention may or may not have been set at the MERLIN beamline of the ALS and may need
    to be changed accordingly in the future.

    `find_ses_files_associated` will collect all the files in the sequence
    pointed to by `reference_path`.

    Args:
        reference_path: One path among the sequence of paths which should be used as the template.
        separator: A separator between the scan number and frame number of each scan.

    Returns:
        The list of files which are associated to a given scan, including `reference_path`.
    """
    name_match = re.match(
        r"([\w+]+)[{}][0-9][0-9][0-9]\.pxt".format(separator), reference_path.name
    )

    if name_match is None:
        return [reference_path]

    # otherwise need to collect all of the components
    fragment = name_match.groups()[0]
    components = list(reference_path.parent.glob("{}*.pxt".format(fragment)))
    components.sort()

    return components


def read_separated_pxt(
    reference_path: Path, separator=None, byte_order: Optional[str] = None
) -> DataType:
    """Reads a series of .pxt files which correspond to cuts in a multi-cut scan.

    Args:
        reference_path: The path to one of the files in the series.
        separator: Unused, but kept in order to preserve uniform API. Defaults to None.
        byte_order: The byte order on the file that is to be read. One of "<", ">", "=".
          If None is provided byte orders will be tried in sequence. Defaults to None.

    Returns:
        Concatenated data corresponding to the waves in the different .pxt files.
    """
    # determine if separated or not
    components = find_ses_files_associated(reference_path)
    frames = [read_single_pxt(f, byte_order=byte_order) for f in components]

    if len(frames) == 1:
        return frames[0]

    # adjust as needed
    scan_coords = ["hv", "polar", "timed_power", "tilt", "volts"]

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
