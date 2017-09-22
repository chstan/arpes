"""
Module: view.py

Exports:
class View

A View into a dataset is a way of handling conversions to different units.
Whereas datasets are recorded in anglespace units and units of energy typically,
it is most natural to do analysis a k-space volume.

A View belongs to a dataset which it represents, but after creation can be
independently analyzed. A second View cannot be created on top of a first,
but because each view keeps of a record of the original spectrum that it was
interpolated from, this is not an issue in practice.

In order to specify a View you need to provide a spectrum, as well as a
description of the axes you would like. Each axis has a 'name', a 'display_name',
and a 'range'.

- 'name' carries semantics related to physical units, and is used to infer the
  appropriate conversion functions to use for constructing a particular View.

- 'display_name' only indicates the name as it should appear on figures for
  data exploration

- 'range' can be either a string, 'original' indicating that the axis should
  emulate the orignal datasets axis of equivalent units with the same spacing
  and bounds, or it can be a dictionary with keys 'bounds' and 'spacing'. For
  full bounds range the 'bounds' key can correspond to a value 'full'.

The View class also provides class attributes providing convenient and commonly
used sets of axes. These also furnish examples of proper use when creating a
view.
"""

import itertools

import numpy
from scipy.interpolate import RegularGridInterpolator

from arpes.exceptions import AnalysisError
from arpes.utilities import arrange_by_indices
from arpes.utilities.conversion import (
    # photon energy scans
    kx_kz_E_to_polar,
    kx_kz_E_to_hv,
    polar_hv_E_to_kx,
    polar_hv_E_to_kz,
    polar_hv_E_corners_to_kx_kz_E_bounds,
    jacobian_polar_hv_E_to_kx_kz_E,

    # tilt geometry
    kx_ky_KE_to_polar,
    kx_ky_KE_to_elev,
    polar_elev_KE_to_kx,
    polar_elev_KE_to_ky,
    polar_elev_KE_corners_to_kx_ky_KE_bounds,
    jacobian_polar_elev_KE_to_kx_ky_KE,
)
from .viewable import Viewable

_coordinate_translations = {
    # Organized by old -> new,
    # which is another way of saying original spectrum axes -> new axes
    (('detector_angle', 'hv', 'KE'),('kx','kz','KE'),): {
        'detector_angle': kx_kz_E_to_polar,
        'hv': kx_kz_E_to_hv,
        'KE': lambda x, y, KE, metadata: KE,
        'kx': polar_hv_E_to_kx,
        'kz': polar_hv_E_to_kz,
        'bounds_from_corners': polar_hv_E_corners_to_kx_kz_E_bounds,
        'jacobian': jacobian_polar_hv_E_to_kx_kz_E,
    },
    # This coordinate translation set is used for the vertical slit orientations
    # with moving analyzers, such as is used at BL10
    (('detector_angle', 'detector_sweep_angle', 'KE'),('kx','ky','KE',)): {
        'detector_angle': kx_ky_KE_to_polar,
        'detector_sweep_angle': kx_ky_KE_to_elev,
        'KE': lambda x, y, KE, metadata: KE,
        'kx': polar_elev_KE_to_kx,
        'ky': polar_elev_KE_to_ky,
        'bounds_from_corners': polar_elev_KE_corners_to_kx_ky_KE_bounds,
        'jacobian': jacobian_polar_elev_KE_to_kx_ky_KE,
    },
}

def get_compatible_coordinate_translation(desired_axes, old_axes):
    for labels, translation in _coordinate_translations.items():
        old_labels, new_labels = labels
        if set(desired_axes) == set(new_labels) and set(old_axes) == set(old_labels):
            # found an appropriate translation,
            # now need to calculate the axis reordering and return the translation
            new_axis_reordering = tuple(desired_axes.index(l) for l in new_labels)
            old_axis_reordering = tuple(old_axes.index(l) for l in old_labels)
            return old_axis_reordering, new_axis_reordering, translation
    return None

_LOW_KX = {
    'name': 'kx',
    'display_name': 'kx',
    'range': {'bounds': 'full', 'spacing': 0.02}
}
_LOW_KY = {
    'name': 'ky',
    'display_name': 'ky',
    'range': {'bounds': 'full', 'spacing': 0.02}
}
_LOW_KZ = {
    'name': 'kz',
    'display_name': 'kz',
    'range': {'bounds': 'full', 'spacing': 0.02}
}

_MEDIUM_KX = {
    'name': 'kx',
    'display_name': 'kx',
    'range': {'bounds': 'full', 'spacing': 0.01}
}
_MEDIUM_KY = {
    'name': 'ky',
    'display_name': 'ky',
    'range': {'bounds': 'full', 'spacing': 0.01}
}
_MEDIUM_KZ = {
    'name': 'kz',
    'display_name': 'kz',
    'range': {'bounds': 'full', 'spacing': 0.01}
}

_HIGH_KX = {
    'name': 'kx',
    'display_name': 'kx',
    'range': {'bounds': 'full', 'spacing': 0.005}
}
_HIGH_KY = {
    'name': 'ky',
    'display_name': 'ky',
    'range': {'bounds': 'full', 'spacing': 0.005}
}
_HIGH_KZ = {
    'name': 'kz',
    'display_name': 'kz',
    'range': {'bounds': 'full', 'spacing': 0.005}
}

_ORIGINAL_E = {
    'name': 'BE',
    'display_name': 'Binding energy',
    'range': 'original'
}

_ORIGINAL_KE = {
    'name': 'KE',
    'display_name': 'Kinetic energy',
    'range': 'original'
}

class View(Viewable):
    # Standard geometry
    LOW_RES_KX_KZ_KE = (_LOW_KX, _LOW_KZ, _ORIGINAL_KE,)
    MEDIUM_RES_KX_KZ_KE = (_MEDIUM_KX, _MEDIUM_KZ, _ORIGINAL_KE,)
    HIGH_RES_KX_KZ_KE = (_HIGH_KX, _HIGH_KZ, _ORIGINAL_KE,)

    # Photon energy scan (tilt geometry)
    LOW_RES_KX_KY_KE = (_LOW_KX, _LOW_KY, _ORIGINAL_KE,)
    MEDIUM_RES_KX_KY_KE = (_MEDIUM_KX, _MEDIUM_KY, _ORIGINAL_KE,)
    HIGH_RES_KX_KY_KE = (_MEDIUM_KX, _MEDIUM_KY, _ORIGINAL_KE,)

    # polar angle in KSpace_JD refers to scanning angle alpha at BL10
    def calculate_own_bounds(self):
        spectrum_corner_points = itertools.product(*[b[:2] for b in self.parent.bounds])
        spectrum_corner_points = [arrange_by_indices(p, self.parent_axis_rearrangement)
                                  for p in spectrum_corner_points]

        # next we need to apply any axis rearrangement
        return self.translation['bounds_from_corners'](
            spectrum_corner_points, metadata=self.parent.metadata)

    def build_interpolator_from_parent(self):
        """
        It's important that the data we get from the parent spectrum is regularly
        spaced. If this were not the case, we would require
        'scipy.interpolate.LinearNDInterpolator, which has to build a Delaunay
        triangulation of the domain to provide lookup information for the
        interpolation.
        """

        # need to put parent ticks and axes in the right order
        swapped_axis_ticks = arrange_by_indices(self.parent.ticks,
                                                self.parent_axis_rearrangement)
        swapped_data = numpy.transpose(self.parent.data, axes=self.parent_axis_rearrangement)
        return RegularGridInterpolator(
            points=swapped_axis_ticks,
            values=swapped_data,
            method="linear",
            fill_value=0.0,
            bounds_error=False)

    def __init__(self, spectrum=None, axes=None, bounds=None, jacobian_corrections=False):
        super(View, self).__init__(data=None, axes=axes, bounds=bounds)

        self.parent = spectrum
        parent_axis_reo, axis_reo, translation = get_compatible_coordinate_translation(
            self.axis_names, self.parent.axis_names)
        self.translation = translation
        self.parent_axis_rearrangement = parent_axis_reo
        self.axis_rearrangement = axis_reo

        if self.translation is None:
            raise AnalysisError('No compatible change of coordinates found for ' +
                                '{}->{}'.format(self.parent.axis_names,
                                                self.axis_names))

        self.parent_axes = spectrum.axes

        parent_shape = self.parent.data.shape

        own_bounds = self.calculate_own_bounds()

        # Next we have to calculate the dimensions of the new view of the data
        own_shape = []
        for i, bound in enumerate(own_bounds):
            axis_info = self.axes[i]
            if axis_info.get('range', 'original') == 'original':
                # Request to use the same number of points as the original axis,
                # this is typically only used for energy or another unmodified axis

                own_shape.append(parent_shape[i])

            else:
                # Need to infer number of points from the requested spacing
                low, high = bound
                own_shape.append(int(
                    (high - low)/axis_info['range']['spacing'] + 1))

        own_shape = tuple(own_shape)

        grid_interpolator = self.build_interpolator_from_parent()
        self.interpolator = grid_interpolator
        self.bounds = [tuple(itertools.chain(b, [s]))
                       for b, s in zip(own_bounds, own_shape)]

        for i, axis in enumerate(self.axes):
            axis['bounds'] = self.bounds[i]

        Xs = numpy.linspace(own_bounds[0][0], own_bounds[0][1], own_shape[0])
        Ys = numpy.linspace(own_bounds[1][0], own_bounds[1][1], own_shape[1])
        Zs = numpy.linspace(own_bounds[2][0], own_bounds[2][1], own_shape[2])

        Xgs, Ygs, Zgs = numpy.meshgrid(Xs, Ys, Zs, indexing='ij')
        Xgs, Ygs, Zgs = Xgs.ravel(), Ygs.ravel(), Zgs.ravel()

        coordinate_translations = arrange_by_indices(
            [translation[a] for a in self.parent.axis_names],
            self.parent_axis_rearrangement)
        self.data = grid_interpolator(numpy.array(
            [coordinate_translations[0](Xgs, Ygs, Zgs, metadata=self.parent.metadata),
             coordinate_translations[1](Xgs, Ygs, Zgs, metadata=self.parent.metadata),
             coordinate_translations[2](Xgs, Ygs, Zgs, metadata=self.parent.metadata)]).T)
        self.raw_data = self.data
        self.data = numpy.reshape(self.data, own_shape, order='C')
