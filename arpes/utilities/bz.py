import numpy as np
import matplotlib.path

__all__ = ['bz_symmetry', 'bz_cutter', 'reduced_bz_selection',
           'reduced_bz_axes', 'reduced_bz_mask', 'reduced_bz_poly',
           'reduced_bz_axis_to', 'reduced_bz_E_mask', 'axis_along']


_SYMMETRY_TYPES = {
    ('G', 'X', 'Y'): 'rect',
    ('G', 'X'): 'square',
    ('G', 'X', 'BX'): 'hex',
}

_POINT_NAMES_FOR_SYMMETRY = {
    'rect': {'G', 'X', 'Y'},
    'square': {'G', 'X'},
    'hex': {'G', 'X', 'BX'},
}

def bz_symmetry(flat_symmetry_points):
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
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v.get(d, 0) for d in data.dims if d in v.keys() or include_E and d == 'eV'])
                       for k, v in points.items()}
    if symmetry == 'rect':
        if S == 'X':
            return coords_by_point['X'] - coords_by_point['G']
        return coords_by_point['Y'] - coords_by_point['G']
    elif symmetry == 'square':
        assert (False)  # FIXME to use other BZ point
        return coords_by_point['X'] - coords_by_point['G']
    elif symmetry == 'hex':
        if S == 'X':
            return coords_by_point['X'] - coords_by_point['G']
        return coords_by_point['BX'] - coords_by_point['G']
    else:
        raise NotImplementedError()


def reduced_bz_axes(data):
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v[d] for d in data.dims if d in v.keys()]) for k, v in points.items()}
    if symmetry == 'rect':
        dx = coords_by_point['X'] - coords_by_point['G']
        dy = coords_by_point['Y'] - coords_by_point['G']
    elif symmetry == 'square':
        assert (False)  # FIXME to use other BZ point
        dx = coords_by_point['X'] - coords_by_point['G']
        dy = coords_by_point['X'] - coords_by_point['G']
    elif symmetry == 'hex':
        dx = coords_by_point['X'] - coords_by_point['G']
        dy = coords_by_point['BX'] - coords_by_point['G']
    else:
        raise NotImplementedError()

    return dx, dy


def axis_along(data, S):
    """
    Determines which axis lies principally along the direction G->S.
    :param data:
    :param S:
    :return:
    """
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v[d] for d in data.dims if d in v.keys()]) for k, v in points.items()}

    dS = coords_by_point[S] - coords_by_point['G']

    max = -np.inf
    max_dim = None
    for dD, d in zip(dS, [d for d in data.dims if d != 'eV']):
        if np.abs(dD) > max:
            max = np.abs(dD)
            max_dim = d

    return max_dim


def reduced_bz_poly(data, scale_zone=False):
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]
    dx, dy = reduced_bz_axes(data)
    if scale_zone:
        # should be good enough, reevaluate later
        dx = 3 * dx
        dy = 3 * dy

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {k: np.array([v.get(d, 0) for d in data.dims if d in v.keys()]) for k, v in points.items()}

    if symmetry == 'hex':
        return np.array([
            coords_by_point['G'],
            coords_by_point['G'] + dx,
            coords_by_point['G'] + dy,
        ])

    return np.array([
        coords_by_point['G'],
        coords_by_point['G'] + dx,
        coords_by_point['G'] + dx + dy,
        coords_by_point['G'] + dy,
    ])


def reduced_bz_E_mask(data, S, e_cut, scale_zone=False):
    symmetry_points, _ = data.S.symmetry_points()
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]
    bz_dims = tuple(d for d in data.dims if d in list(symmetry_points.values())[0][0].keys())


    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {k: np.array([v.get(d,0) for d in data.dims if d in v.keys() or d == 'eV'])
                       for k, v in points.items()}

    dx_to = reduced_bz_axis_to(data, S, include_E=True)
    if scale_zone:
        dx_to = dx_to * 3
    dE = np.array([0 if d != 'eV' else e_cut for d in data.dims])

    poly_points = np.array([
        coords_by_point['G'],
        coords_by_point['G'] + dx_to,
        coords_by_point['G'] + dx_to + dE,
        coords_by_point['G'] + dE,
    ])

    skip_col = None
    for i in range(poly_points.shape[1]):
        if np.all(poly_points[:, i] == poly_points[0, i]):
            skip_col = i

    assert(skip_col is not None)
    selector_val = poly_points[0, skip_col]
    poly_points = np.concatenate((poly_points[:, 0:skip_col], poly_points[:, skip_col+1:]), axis=1)

    selector = dict()
    selector[data.dims[skip_col]] = selector_val
    sdata = data.sel(**selector, method='nearest')

    path = matplotlib.path.Path(poly_points)
    grid = np.array([a.ravel() for a in np.meshgrid(*[data.coords[d] for d in sdata.dims], indexing='ij')]).T
    mask = path.contains_points(grid)
    mask = np.reshape(mask, sdata.data.shape)
    return mask


def reduced_bz_mask(data, **kwargs):
    symmetry_points, _ = data.S.symmetry_points()
    bz_dims = tuple(d for d in data.dims if d in list(symmetry_points.values())[0][0].keys())

    poly_points = reduced_bz_poly(data, **kwargs)
    extra_dims_shape = tuple(len(data.coords[d]) for d in data.dims
                             if d in bz_dims)

    path = matplotlib.path.Path(poly_points)
    grid = np.array([a.ravel() for a in np.meshgrid(*[data.coords[d] for d in bz_dims], indexing='ij')]).T
    mask = path.contains_points(grid)
    mask = np.reshape(mask, extra_dims_shape)

    return mask


def reduced_bz_selection(data):
    mask = reduced_bz_mask(data)

    data = data.copy()
    data.data[np.logical_not(mask)] = np.nan

    return data


def bz_cutter(symmetry_points, reduced=True):
    """
    TODO UNFINISHED
    :param symmetry_points:
    :param reduced:
    :return:
    """
    def build_bz_mask(data):
        pass



    def cutter(data, cut_value=np.nan):
        mask = build_bz_mask(data)

        out = data.copy()
        out.data[mask] = cut_value

        return out

    return cutter