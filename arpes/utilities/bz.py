"""Utilities related to computing Brillouin zones, masks, and selecting data.

TODO: Standardize this module around support for some other library that has proper
Brillouin zone plotting, like in ASE.

This module also includes tools for masking regions of data against
Brillouin zones.
"""

import itertools
import re
from collections import Counter, namedtuple

import matplotlib.path
import numpy as np

__all__ = (
    "bz_symmetry",
    "bz_cutter",
    "reduced_bz_selection",
    "reduced_bz_axes",
    "reduced_bz_mask",
    "reduced_bz_poly",
    "reduced_bz_axis_to",
    "reduced_bz_E_mask",
    "axis_along",
    "hex_cell",
    "hex_cell_2d",
    "orthorhombic_cell",
    "process_kpath",
)


_SYMMETRY_TYPES = {
    ("G", "X", "Y"): "rect",
    ("G", "X"): "square",
    ("G", "X", "BX"): "hex",
}

_POINT_NAMES_FOR_SYMMETRY = {
    "rect": {"G", "X", "Y"},
    "square": {"G", "X"},
    "hex": {"G", "X", "BX"},
}

SpecialPoint = namedtuple("SpecialPoint", ("name", "negate", "bz_coord"))


def as_3d(points):
    """Takes a 2D points list and zero pads to convert to a 3D representation."""
    return np.concatenate([points, points[:, 0][:, None] * 0], axis=1)


def as_2d(points):
    """Takes a 3D points and converts to a 2D representation by dropping the z coordinates."""
    return points[:, :2]


def parse_single_path(path):
    """Converts a path given by high symmetry point names to numerical coordinate arrays."""
    # first tokenize
    tokens = [name for name in re.split(r"([A-Z][a-z0-9]*(?:\([0-9,\s]+\))?)", path) if name]

    # normalize Gamma to G
    tokens = [token.replace("Gamma", "G") for token in tokens]

    # convert to standard format
    points = []
    for token in tokens:
        name, rest = token[0], token[1:]
        negate = False
        if rest and rest[0] == "n":
            negate = True
            rest = rest[1:]

        bz_coords = (
            0,
            0,
            0,
        )
        if rest:
            rest = "".join(c for c in rest if c not in "( \t\n\r)")
            bz_coords = tuple([int(c) for c in rest.split(",")])

        if len(bz_coords) == 2:
            bz_coords = tuple(list(bz_coords) + [0])
        points.append(SpecialPoint(name=name, negate=negate, bz_coord=bz_coords))

    return points


def parse_path(paths):
    """Converts paths to arrays with the coordinate locations for those paths."""
    if isinstance(paths, str):

        # some manual string work in order to make sure we do not split on commas inside BZ indices
        idxs = []
        for i, p in enumerate(paths):
            if p == ",":
                c = Counter(paths[:i])
                if c["("] - c[")"] == 0:
                    idxs.append(i)

        paths = list(paths)
        for idx in idxs:
            paths[idx] = ":"

        paths = "".join(paths)
        paths = paths.split(":")

    return [parse_single_path(p) for p in paths]


def special_point_to_vector(special_point, icell, special_points):
    """Converts a single special point to its coordinate vector."""
    base = np.dot(icell.T, special_points[special_point.name])

    if special_point.negate:
        base = -np.array(base)

    coord = np.array(special_point.bz_coord)
    return base + coord.dot(icell)


def process_kpath(paths, cell, special_points=None):
    """Converts paths consistign of point definitions to raw coordinates."""
    if len(cell) == 2:
        cell = [c + [0] for c in cell] + [0, 0, 0]

    icell = np.linalg.inv(cell).T

    if special_points is None:
        from ase.dft.kpoints import get_special_points

        special_points = get_special_points(cell)

    points = [
        [special_point_to_vector(elem, icell, special_points) for elem in p]
        for p in parse_path(paths)
    ]

    return points


# Some common Brillouin zone formats
def orthorhombic_cell(a=1, b=1, c=1):
    """Lattice constants for an orthorhombic unit cell."""
    return [[a, 0, 0], [0, b, 0], [0, 0, c]]


def hex_cell(a=1, c=1):
    """Calculates lattice vectors for a triangular lattice with lattice constants `a` and `c`."""
    return [[a, 0, 0], [-0.5 * a, 3 ** 0.5 / 2 * a, 0], [0, 0, c]]


def hex_cell_2d(a=1):
    """Calculates lattice vectors for a triangular lattice with lattice constant `a`."""
    return [[a, 0], [-0.5 * a, 3 ** 0.5 / 2 * a]]


def flat_bz_indices_list(bz_indices_list=None):
    """Calculate a flat representation of a repeated Brillouin zone specification.

    This is useful for plotting extra Brillouin zones or generating high symmetry points,
    lines, and planes.

    If None is provided, the first BZ is assumed.

    ```
    None -> [(0,0)]
    ```

    If an explicit zone is provided or a list of zones is provided, these are
    returned

    ```
    [(0,1,0), (-1, -1, 2)] -> [(0,1,0), (-1, -1, 2)]
    ```

    Additionally, tuples are unpacked into ranges

    ```
    [((-2, 1), 1)] -> [(-2, 1), (-1, 1), (0, 1)]
    ```
    """
    if bz_indices_list is None:
        bz_indices_list = [(0, 0)]

    try:
        l = len(bz_indices_list[0])
        if l not in {2, 3}:
            raise ValueError()
    except (ValueError, TypeError):
        bz_indices_list = [bz_indices_list]

    indices = []
    if len(bz_indices_list[0]) == 2:
        for bz_x, bz_y in bz_indices_list:
            rx = range(bz_x, bz_x + 1) if isinstance(bz_x, int) else range(*bz_x)
            ry = range(bz_y, bz_y + 1) if isinstance(bz_y, int) else range(*bz_y)
            for x, y in itertools.product(rx, ry):
                indices.append(
                    (
                        x,
                        y,
                    )
                )
    else:
        for bz_x, bz_y, bz_z in bz_indices_list:
            rx = range(bz_x, bz_x + 1) if isinstance(bz_x, int) else range(*bz_x)
            ry = range(bz_y, bz_y + 1) if isinstance(bz_y, int) else range(*bz_y)
            rz = range(bz_z, bz_z + 1) if isinstance(bz_z, int) else range(*bz_z)
            for x, y, z in itertools.product(rx, ry, rz):
                indices.append((x, y, z))

    return indices


def generate_2d_equivalent_points(points, icell, bz_indices_list=None):
    """Generates the equivalent points in higher order Brillouin zones."""
    points_list = []
    for x, y in flat_bz_indices_list(bz_indices_list):
        points_list.append(
            points[:, :2]
            + x
            * icell[0][
                None,
                :2,
            ]
            + y
            * icell[1][
                None,
                :2,
            ]
        )

    return np.unique(np.concatenate(points_list), axis=0)


def build_2dbz_poly(vertices=None, icell=None, cell=None):
    """Converts brillouin zone or equivalent information to a polygon mask.

    This mask can be used to mask away data outside the zone boundary.
    """
    from arpes.analysis.mask import raw_poly_to_mask
    from ase.dft.bz import bz_vertices  # pylint: disable=import-error

    assert cell is not None or vertices is not None or icell is not None

    if vertices is None:
        if icell is None:
            icell = np.linalg.inv(cell).T

        vertices = bz_vertices(icell)

    points, _ = vertices[0]  # points, normal
    points_2d = [p[:2] for p in points]

    return raw_poly_to_mask(points_2d)


def bz_symmetry(flat_symmetry_points):
    """Determines symmetry from a list of the symmetry points."""
    if isinstance(flat_symmetry_points, dict):
        flat_symmetry_points = flat_symmetry_points.items()

    largest_identified = 0
    symmetry = None

    point_names = set(k for k, _ in flat_symmetry_points)

    for points, sym in _SYMMETRY_TYPES.items():
        if all(p in point_names for p in points):
            if len(points) > largest_identified:
                symmetry = sym
                largest_identified = len(points)

    return symmetry


def reduced_bz_axis_to(data, S, include_E=False):
    """Calculates a displacement vector to a modded high symmetry point."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v.keys() or include_E and d == "eV"])
        for k, v in points.items()
    }
    if symmetry == "rect":
        if S == "X":
            return coords_by_point["X"] - coords_by_point["G"]
        return coords_by_point["Y"] - coords_by_point["G"]
    elif symmetry == "square":
        raise NotImplementedError
        return coords_by_point["X"] - coords_by_point["G"]
    elif symmetry == "hex":
        if S == "X":
            return coords_by_point["X"] - coords_by_point["G"]
        return coords_by_point["BX"] - coords_by_point["G"]
    else:
        raise NotImplementedError


def reduced_bz_axes(data):
    """Calculates displacement vectors to high symmetry points in the first Brillouin zone."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {
        k: np.array([v[d] for d in data.dims if d in v.keys()]) for k, v in points.items()
    }
    if symmetry == "rect":
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["Y"] - coords_by_point["G"]
    elif symmetry == "square":
        raise NotImplementedError
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["X"] - coords_by_point["G"]
    elif symmetry == "hex":
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["BX"] - coords_by_point["G"]
    else:
        raise NotImplementedError

    return dx, dy


def axis_along(data, S):
    """Determines which axis lies principally along the direction G->S."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {
        k: np.array([v[d] for d in data.dims if d in v.keys()]) for k, v in points.items()
    }

    dS = coords_by_point[S] - coords_by_point["G"]

    max = -np.inf
    max_dim = None
    for dD, d in zip(dS, [d for d in data.dims if d != "eV"]):
        if np.abs(dD) > max:
            max = np.abs(dD)
            max_dim = d

    return max_dim


def reduced_bz_poly(data, scale_zone=False):
    """Returns a polynomial representing the reduce first Brillouin zone."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]
    dx, dy = reduced_bz_axes(data)
    if scale_zone:
        # should be good enough, reevaluate later
        dx = 3 * dx
        dy = 3 * dy

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v.keys()]) for k, v in points.items()
    }

    if symmetry == "hex":
        return np.array(
            [
                coords_by_point["G"],
                coords_by_point["G"] + dx,
                coords_by_point["G"] + dy,
            ]
        )

    return np.array(
        [
            coords_by_point["G"],
            coords_by_point["G"] + dx,
            coords_by_point["G"] + dx + dy,
            coords_by_point["G"] + dy,
        ]
    )


def reduced_bz_E_mask(data, S, e_cut, scale_zone=False):
    """Calculates a mask for data which contains points below an energy cutoff."""
    symmetry_points, _ = data.S.symmetry_points()
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]
    # bz_dims = tuple(d for d in data.dims if d in list(symmetry_points.values())[0][0].keys())

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v.keys() or d == "eV"])
        for k, v in points.items()
    }

    dx_to = reduced_bz_axis_to(data, S, include_E=True)
    if scale_zone:
        dx_to = dx_to * 3
    dE = np.array([0 if d != "eV" else e_cut for d in data.dims])

    poly_points = np.array(
        [
            coords_by_point["G"],
            coords_by_point["G"] + dx_to,
            coords_by_point["G"] + dx_to + dE,
            coords_by_point["G"] + dE,
        ]
    )

    skip_col = None
    for i in range(poly_points.shape[1]):
        if np.all(poly_points[:, i] == poly_points[0, i]):
            skip_col = i

    assert skip_col is not None
    selector_val = poly_points[0, skip_col]
    poly_points = np.concatenate(
        (poly_points[:, 0:skip_col], poly_points[:, skip_col + 1 :]), axis=1
    )

    selector = dict()
    selector[data.dims[skip_col]] = selector_val
    sdata = data.sel(**selector, method="nearest")

    path = matplotlib.path.Path(poly_points)
    grid = np.array(
        [a.ravel() for a in np.meshgrid(*[data.coords[d] for d in sdata.dims], indexing="ij")]
    ).T
    mask = path.contains_points(grid)
    mask = np.reshape(mask, sdata.data.shape)
    return mask


def reduced_bz_mask(data, **kwargs):
    """Calculates a mask for the first Brillouin zone of a piece of data."""
    symmetry_points, _ = data.S.symmetry_points()
    bz_dims = tuple(d for d in data.dims if d in list(symmetry_points.values())[0][0].keys())

    poly_points = reduced_bz_poly(data, **kwargs)
    extra_dims_shape = tuple(len(data.coords[d]) for d in data.dims if d in bz_dims)

    path = matplotlib.path.Path(poly_points)
    grid = np.array(
        [a.ravel() for a in np.meshgrid(*[data.coords[d] for d in bz_dims], indexing="ij")]
    ).T
    mask = path.contains_points(grid)
    mask = np.reshape(mask, extra_dims_shape)

    return mask


def reduced_bz_selection(data):
    """Sets data outisde the Brillouin zone mask for a piece of data to be nan."""
    mask = reduced_bz_mask(data)

    data = data.copy()
    data.data[np.logical_not(mask)] = np.nan

    return data


def bz_cutter(symmetry_points, reduced=True):
    """Cuts data so that it areas outside the Brillouin zone are masked away.

    TODO: UNFINISHED.
    """

    def build_bz_mask(data):
        pass

    def cutter(data, cut_value=np.nan):
        mask = build_bz_mask(data)

        out = data.copy()
        out.data[mask] = cut_value

        return out

    return cutter
