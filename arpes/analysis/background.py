"""Provides background estimation approaches."""
import xarray as xr
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

__all__ = (
    "calculate_background_hull",
    "remove_background_hull",
)


def calculate_background_hull(arr, breakpoints=None):
    """Calculates a background using the convex hull of the data (viewing the intensity as a Z axis)."""
    if breakpoints:
        breakpoints = [None] + breakpoints + [None]
        dim = arr.dims[0]
        processed = []
        for blow, bhigh in zip(breakpoints, breakpoints[1:]):
            processed.append(
                calculate_background_hull(arr.sel(**dict([[dim, slice(blow, bhigh)]])))
            )
        return xr.concat(processed, dim)

    points = np.stack(arr.G.to_arrays(), axis=1)
    hull = ConvexHull(points)

    vertices = np.array(hull.vertices)
    index_of_zero = np.argwhere(vertices == 0)[0][0]
    vertices = np.roll(vertices, -index_of_zero)
    xis = vertices[: np.argwhere(vertices == len(arr) - 1)[0][0]]
    xis = list(xis) + [len(arr) - 1]

    support = points[xis]
    bkg = interp1d(support[:, 0], support[:, 1], fill_value="extrapolate")(points[:, 0])
    return arr.S.with_values(interp1d(support[:, 0], support[:, 1])(points[:, 0]))


def remove_background_hull(data, *args, **kwargs):
    """Removes a background according to `calculate_background_hull`."""
    return data - calculate_background_hull(data, *args, **kwargs)
