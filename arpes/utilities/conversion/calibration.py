"""Preliminary detector window corrections."""
import scipy.interpolate
import xarray as xr
import numpy as np
from typing import List, Dict
import itertools

__all__ = ("DetectorCalibration",)


def build_edge_from_list(points: List[Dict[str, float]]):
    """Converts from a list of edge waypoints to a common representation as a DataSet."""
    dimensions = set(itertools.chain(*[p.keys() for p in points]))
    arrays = {}

    for dim in dimensions:
        values = [p[dim] for p in points]
        arrays[dim] = xr.DataArray(values, coords=dict([[dim, values]]), dims=[dim])
    return xr.Dataset(arrays)


class DetectorCalibration:
    """A detector calibration model allowing for correcting the trapezoidal windowing."""

    _left_edge: xr.Dataset
    _right_edge: xr.Dataset

    def __init__(self, left, right):
        """Build the edges for the calibration from a path for the left and right sides."""
        assert set(left[0].keys()) == {
            "phi",
            "eV",
        }
        self._left_edge = build_edge_from_list(left)
        self._right_edge = build_edge_from_list(right)

        # if the edges were passed incorrectly then do it ourselves
        if self._left_edge.phi.mean() > self._right_edge.phi.mean():
            self._left_edge, self._right_edge = self._right_edge, self._left_edge

    def __repr__(self):
        """Representation showing detailed attributes on edge locations."""
        rep = f"<DetectorCalibration>\n\n"
        rep += "Left Edge\n"
        rep += str(self._left_edge)
        rep += "\n\nRightEdge\n"
        rep += str(self._right_edge)
        return rep

    def correct_detector_angle(self, eV, phi):
        """Applies a calibration to the detector `phi` angle."""
        left, right = (
            scipy.interpolate.interp1d(self._left_edge.eV.values, self._left_edge.phi.values)(
                0
            ).item(),
            scipy.interpolate.interp1d(self._right_edge.eV.values, self._right_edge.phi.values)(
                0
            ).item(),
        )
        xs = np.concatenate([self._left_edge.eV.values, self._right_edge.eV.values])
        ys = np.concatenate([self._left_edge.phi.values, self._right_edge.phi.values])
        zs = np.concatenate(
            [self._left_edge.eV.values * 0 + left, self._right_edge.eV.values * 0 + right]
        )

        return scipy.interpolate.griddata(np.stack([xs, zs], axis=1), ys, (eV, phi))
