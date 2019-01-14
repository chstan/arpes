import xarray as xr
import numpy as np

from arpes.typing import DataType

__all__ = ('discretize_path', 'select_around_path', 'path_from_points',)


def path_from_points(data: DataType, symmetry_points_or_interpolation_points):
    """
    Acceepts a list of either tuples or point references. Point references can be string keys to `.attrs['symmetry_points']`
    This is the same behavior as `analysis.slice_along_path` and underlies the logic there.
    :param data:
    :param symmetry_points_or_interpolation_points:
    :return:
    """
    raise NotImplementedError('')


def discretize_path(path: xr.Dataset, n_points=None, scaling=None):
    """
    Shares logic with slice_along_path
    :param path:
    :param n_points:
    :return:
    """

    if scaling is None:
        scaling = 1
    elif isinstance(scaling, xr.Dataset):
        scaling = {k: scaling[k].item() for k in scaling.data_vars}
    else:
        assert isinstance(scaling, dict)

    order = list(path.data_vars)
    if isinstance(scaling, dict):
        scaling = np.array(scaling[d] for d in order)

    def as_vec(ds):
        return np.array(list(ds[k].item() for k in order))

    def distance(a, b):
        return np.linalg.norm((as_vec(a) - as_vec(b)) * scaling)

    length = 0
    for idx_low, idx_high in (zip(path.index.values, path.index[1:].values)):
        coord_low, coord_high = path.sel(index=idx_low), path.sel(index=idx_high)
        length += distance(coord_low, coord_high)

    if n_points is None:
        # play with this until it seems reasonable
        n_points = int(length / 0.03)
    else:
        n_points = max(n_points - 1, 1)

    sep = length / n_points
    points = []
    distances = np.linspace(0, n_points - 1, n_points) * sep

    total_dist = 0
    for idx_low, idx_high in (zip(path.index.values, path.index[1:].values)):
        coord_low, coord_high = path.sel(index=idx_low), path.sel(index=idx_high)

        current_dist = distance(coord_low, coord_high)
        current_points = distances[distances < total_dist + current_dist]
        current_points = (current_points - total_dist) / current_dist
        distances = distances[len(current_points):]
        total_dist += current_dist

        points = points + list(np.outer(current_points, as_vec(coord_high) - as_vec(coord_low)) + as_vec(coord_low))

    points.append(as_vec(path.sel(index=path.index.values[-1])))

    new_index = np.array(range(len(points)))

    def to_dataarray(name):
        index = order.index(name)
        data = [p[index] for p in points]

        return xr.DataArray(np.array(data), {'index': new_index}, ['index'])

    return xr.Dataset({k: to_dataarray(k) for k in order})


def select_along_path(path: xr.Dataset, data: DataType, radius=None, n_points=None, fast=True, scaling=None, **kwargs):
    """
    Performs integration along the path des
    :param path:
    :param data:
    :param radius: A number or dictionary of radii to use for the selection along different dimensions, if none is provided
    reasonable values will be chosen. Alternatively, you can pass radii via `{dim}_r` kwargs as well, i.e. 'eV_r' or 'kp_r'
    :param n_points: The number of points to interpolate along the path, by default we will infer a reasonable number
    from the radius parameter, if provided or inferred
    :param fast: If fast is true, will use rectangular selections rather than ellipsoid ones
    :return:
    """
    new_path = discretize_path(path, n_points, scaling)

    selections = []
    for _, view in new_path.T.iterate_axis('index'):
        selections.append(data.S.select_around(view, radius=radius, fast=fast, **kwargs))

    return xr.concat(selections, new_path.index)