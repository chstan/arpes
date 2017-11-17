import collections
import copy
import warnings

import numpy as np
import xarray as xr

import arpes.constants
import arpes.materials
from arpes.io import load_dataset_attrs
from arpes.plotting import dispersion, spin
from arpes.plotting import make_bokeh_tool, make_curvature_tool
from arpes.utilities.conversion import slice_along_path

__all__ = ['ARPESDataArrayAccessor', 'ARPESDatasetAccessor']


class _ARPESAccessorBase(object):
    """
    Base class for the xarray extensions that we put onto our datasets to make working with ARPES data a
    little cleaner. This allows you to access common attributes
    """

    def along(self, directions):
        return slice_along_path(self._obj, directions)

    @property
    def hv(self):
        if 'hv' in self._obj.attrs:
            value = float(self._obj.attrs['hv'])
            if not np.isnan(value):
                return value

        if 'location' in self._obj.attrs:
            if self._obj.attrs['location'] == 'ALG-MC':
                return 5.93

        return None


    def fetch_ref_attrs(self):
        if 'ref_attrs' in self._obj.attrs:
            return self._obj.attrs
        if 'ref_id' in self._obj.attrs:
            self._obj.attrs['ref_attrs'] = load_dataset_attrs(self._obj.attrs['ref_id'])

        return {}


    @property
    def is_differentiated(self):
        history = self.short_history()
        return 'dn_along_axis' in history or 'curvature' in history

    def short_history(self, key='by'):
        return [h['record'][key] if isinstance(h, dict) else h for h in self.history]

    def _calculate_symmetry_points(self, symmetry_points, projection_distance=0.05,
                                   include_projected=True, epsilon=0.01):
        # For each symmetry point, we need to determine if it is projected or not
        # if it is projected, we need to calculate its projected coordinates
        points = collections.defaultdict(list)
        projected_points = collections.defaultdict(list)

        fixed_coords = {k: v for k, v in self._obj.coords.items() if k not in self._obj.indexes}
        index_coords = self._obj.indexes

        for point, locations in symmetry_points.items():
            if not isinstance(locations, list):
                locations = [locations]

            for location in locations:
                # determine whether the location needs to be projected
                projected = False
                skip = False
                for axis, value in location.items():
                    if axis in fixed_coords and np.abs(value - fixed_coords[axis]) > epsilon:
                        projected = True
                    if axis not in fixed_coords and axis not in index_coords:
                        # cannot even hope to do anything here, we don't have enough info
                        skip = True

                if skip:
                    continue

                new_location = location.copy()
                if projected:
                    # Go and do the projection, for now we will assume we just get it by
                    # replacing the value of the mismatched coordinates.

                    # This does not work if the coordinate system is not orthogonal
                    for axis, v in location.items():
                        if axis in fixed_coords:
                            new_location = fixed_coords[axis]

                    projected_points[point].append(location)
                else:
                    points[point].append(location)

        return points, projected_points

    def symmetry_points(self, **kwargs):
        symmetry_points = self.fetch_ref_attrs().get('symmetry_points', {})
        our_symmetry_points = self._obj.attrs.get('symmetry_points', {})
        copy.deepcopy(symmetry_points)

        symmetry_points.update(our_symmetry_points)
        return self._calculate_symmetry_points(symmetry_points, **kwargs)

    @property
    def history(self):
        provenance = self._obj.attrs.get('provenance', None)

        def unlayer(prov):
            if prov is None:
                return [], None
            if isinstance(prov, str):
                return [prov], None
            first_layer = copy.copy(prov)

            rest = first_layer.pop('parents_provenance', None)
            if rest is None:
                rest = first_layer.pop('parents_provanence', None)
            if isinstance(rest, list):
                warnings.warn('Encountered multiple parents in history extraction, '
                              'throwing away all but the first.')
                if len(rest):
                    rest = rest[0]
                else:
                    rest = None

            return [first_layer], rest

        def _unwrap_provenance(prov):
            if prov is None:
                return []

            first, rest = unlayer(prov)

            return first + _unwrap_provenance(rest)

        return _unwrap_provenance(provenance)


    @property
    def spectrometer(self):
        ds = self._obj
        spectrometers = {
            'SToF': arpes.constants.SPECTROMETER_SPIN_TOF,
            'ToF': arpes.constants.SPECTROMETER_STRAIGHT_TOF,
            'DLD': arpes.constants.SPECTROMETER_DLD,
        }

        if 'spectrometer_name' in ds.attrs:
            return spectrometers.get(ds.attrs['spectrometer_name'])

        if isinstance(ds, xr.Dataset):
            if 'up' in ds.data_vars or ds.attrs.get("18  MCP3") == 0:
                return spectrometers['SToF']
        elif isinstance(ds, xr.DataArray):
            if ds.name == 'up' or ds.attrs.get("18  MCP3") == 0:
                return spectrometers['SToF']

        if 'location' in ds.attrs:
            return {
                'ALG-MC': arpes.constants.SPECTROMETER_MC,
                'BL403': arpes.constants.SPECTROMETER_BL4,
                'ALG-SToF': arpes.constants.SPECTROMETER_STRAIGHT_TOF,
            }.get(ds.attrs['location'])

        try:
            return spectrometers[ds.attrs['spectrometer_name']]
        except KeyError:
            return {}

    @property
    def scan_name(self):
        return self._obj.attrs.get('scan', self._obj.attrs.get('file', self._obj.attrs['id']))

    @property
    def label(self):
        return self._obj.attrs.get('description', self.scan_name)

    @property
    def t0(self):
        if 't0' in self._obj.attrs:
            value = float(self._obj.attrs['t0'])
            if not np.isnan(value):
                return value

        if 'T0_ps' in self._obj.attrs:
            value = float(self._obj.attrs['T0_ps'])
            if not np.isnan(value):
                return value

        return None

    @property
    def polar_offset(self):
        if 'G' in self._obj.attrs.get('symmetry_points', {}):
            gamma_point = self._obj.attrs['symmetry_points']['G']
            if 'polar' in gamma_point:
                return gamma_point['polar']

        if 'polar_offset' in self._obj.attrs:
            return self._obj.attrs['polar_offset']

        return self._obj.attrs.get('data_preparation', {}).get('polar_offset', 0)

    @property
    def phi_offset(self):
        if 'G' in self._obj.attrs.get('symmetry_points', {}):
            gamma_point = self._obj.attrs['symmetry_points']['G']
            if 'phi' in gamma_point:
                return gamma_point['phi']

        if 'polar_offset' in self._obj.attrs:
            return self._obj.attrs['phi_offset']

        return self._obj.attrs.get('data_preparation', {}).get('phi_offset', 0)

    @property
    def material(self):
        try:
            return arpes.materials.material_by_formula[self._obj.attrs['sample']]
        except:
            return None

    @property
    def work_function(self):
        if 'sample_workfunction' in self._obj.attrs:
            return self._obj.attrs['sample_workfunction']

        if self.material:
            return self.material.get('work_function', 4.32)

        return 4.32

    @property
    def inner_potential(self):
        if 'inner_potential' in self._obj.attrs:
            return self._obj.attrs['inner_potential']

        if self.material:
            return self.material.get('inner_potential', 10)

        return 10

    @property
    def sample_pos(self):
        return (self._obj.attrs['x'], self._obj.attrs['y'], self._obj.attrs['z'],)


    @property
    def temp(self):
        warnings.warn('This is not reliable. Fill in stub for normalizing the temperature appropriately on data load.')
        return float(self._obj.attrs['TA'])

    @property
    def fermi_surface(self):
        return self._obj.sel(eV=slice(-0.05, 0.05)).sum('eV', keep_attrs=True)

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataarray_accessor('S')
class ARPESDataArrayAccessor(_ARPESAccessorBase):
    def show(self):
        return make_bokeh_tool(self._obj)

    def show_d2(self):
        return make_curvature_tool(self._obj)

    def fs_plot(self, **kwargs):
        out = kwargs.get('out')
        if out is None and isinstance(out, bool):
            out = '{}_fs.png'.format(self.label)
            kwargs['out'] = out
        return dispersion.labeled_fermi_surface(self._obj, **kwargs)

    def dispersion_plot(self, **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = '{}_dispersion.png'.format(self.label)
            kwargs['out'] = out
        return dispersion.fancy_dispersion(self._obj, **kwargs)

@xr.register_dataset_accessor('S')
class ARPESDatasetAccessor(_ARPESAccessorBase):
    def polarization_plot(self, **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = '{}_spin_polarization.png'.format(self.label)
            kwargs['out'] = out
        return spin.spin_polarized_spectrum(self._obj, **kwargs)


    def __init__(self, xarray_obj):
        super(ARPESDatasetAccessor, self).__init__(xarray_obj)
        self._spectrum = None

        # TODO consider how this should work
        # data_arr_names = self._obj.data_vars.keys()
        #
        # spectrum_candidates = ['raw', 'spectrum', 'spec']
        # if len(data_arr_names) == 1:
        #     self._spectrum = self._obj.data_vars[data_arr_names[0]]
        # else:
        #     for candidate in spectrum_candidates:
        #         if candidate in data_arr_names:
        #             self._spectrum = self._obj.data_vars[candidate]
        #             break
        #
        # if self._spectrum is None:
        #     assert(False and "Dataset appears not to contain a spectrum among options {}".format(data_arr_names))