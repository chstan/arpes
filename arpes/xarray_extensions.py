import collections
import copy
import warnings

import numpy as np
import xarray as xr

import arpes.constants
import arpes.materials
from arpes.io import load_dataset_attrs
import arpes.plotting as plotting
from arpes.plotting import ImageTool, CurvatureTool, BandTool
from arpes.utilities.conversion import slice_along_path

__all__ = ['ARPESDataArrayAccessor', 'ARPESDatasetAccessor']


def _iter_groups(grouped):
    for k, l in grouped.items():
        try:
            for li in l:
                yield k, li
        except TypeError:
            yield k, l


class ARPESAccessorBase(object):
    """
    Base class for the xarray extensions that we put onto our datasets to make working with ARPES data a
    little cleaner. This allows you to access common attributes
    """

    def along(self, directions, **kwargs):
        return slice_along_path(self._obj, directions, **kwargs)

    @property
    def is_kspace(self):
        """
        Infers whether the scan is k-space converted or not. Because of the way this is defined, it will return
        true for XPS spectra, which I suppose is true but trivially.
        :return:
        """

        dims = self._obj.dims
        return not any(d in {'phi', 'polar', 'angle'} for d in dims)

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

        try:
            df = self._obj.attrs['df']
            ref_id = df[df.id == self.original_id].iloc[0].ref_id
            return load_dataset_attrs(ref_id)
        except Exception:
            return {}

    @property
    def spectrum_type(self):
        dim_types = {
            ('eV',): 'xps_spectrum',
            ('eV', 'phi'): 'spectrum',
            ('eV', 'phi', 'polar'): 'map',
            ('eV', 'hv', 'phi'): 'hv_map',

            # kspace
            ('eV', 'kp'): 'spectrum',
            ('eV', 'kx', 'ky'): 'map',
            ('eV', 'kp', 'kz'): 'hv_map',
        }

        dims = tuple(sorted(list(self._obj.dims)))

        return dim_types.get(dims)

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

    def symmetry_points(self, raw=False, **kwargs):
        symmetry_points = self.fetch_ref_attrs().get('symmetry_points', {})
        our_symmetry_points = self._obj.attrs.get('symmetry_points', {})

        symmetry_points = copy.deepcopy(symmetry_points)
        symmetry_points.update(our_symmetry_points)

        if raw:
            return symmetry_points

        return self._calculate_symmetry_points(symmetry_points, **kwargs)

    @property
    def iter_own_symmetry_points(self):
        sym_points, _ = self.symmetry_points()
        return _iter_groups(sym_points)

    @property
    def iter_projected_symmetry_points(self):
        _, sym_points = self.symmetry_points()
        return _iter_groups(sym_points)

    @property
    def iter_symmetry_points(self):
        for sym_point in self.iter_own_symmetry_points:
            yield sym_point
        for sym_point in self.iter_projected_symmetry_points:
            yield sym_point

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
    def dshape(self):
        arr = self._obj
        return dict(zip(arr.dims, arr.shape))

    @property
    def original_id(self):
        history = self.history
        if len(history) >= 3:
            first_modification = history[-3]
            return first_modification['parent_id']

        return self._obj.attrs['id']

    @property
    def original_parent_scan_name(self):
        history = self.history
        if len(history) >= 3:
            first_modification = history[-3]
            df = self._obj.attrs['df']
            return df[df.id == first_modification['parent_id']].index[0]

        return ""

    @property
    def scan_name(self):
        for option in ['scan', 'file']:
            if option in self._obj.attrs:
                return self._obj.attrs[option]

        id = self._obj.attrs.get('id')

        if id is None:
            return 'No ID'

        try:
            df = self._obj.attrs['df']
            return df[df.id == id].index[0]
        except IndexError:
            # data is probably not raw data
            return self.original_parent_scan_name

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

    def _lookup_offset(self, attr_name):
        symmetry_points = self.symmetry_points(raw=True)
        if 'G' in symmetry_points:
            gamma_point = symmetry_points['G']
            if attr_name in gamma_point:
                return gamma_point[attr_name]

        offset_name = attr_name + '_offset'
        if offset_name in self._obj.attrs:
            return self._obj.attrs[offset_name]

        return self._obj.attrs.get('data_preparation', {}).get(offset_name, 0)

    @property
    def polar_offset(self):
        return self._lookup_offset('polar')

    @property
    def phi_offset(self):
        return self._lookup_offset('phi')

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

    def fat_sel(self, widths=None, **kwargs):
        """
        Allows integrating a selection over a given width. The produced dataset
        will be normalized by dividing by the number of single slices integrated over.

        This can be used to produce temporary datasets that have reduced
        uncorrelated noise.
        :param widths: Override the widths for the slices. Resonable defaults are used otherwise
        :param kwargs: slice dict. Has the same function as xarray.DataArray.sel
        :return:
        """
        if widths is None:
            widths = {}

        default_widths = {
            'eV': 0.1,
            'phi': 2,
            'polar': 2,
            'kx': 0.02,
            'ky': 0.02,
            'kp': 0.02,
            'kz': 0.1,
        }

        extra_kwargs = {k: v for k, v in kwargs.items() if k not in self._obj.dims}
        slice_kwargs = {k: v for k, v in kwargs.items() if k not in extra_kwargs}
        slice_widths = {k: widths.get(k, extra_kwargs.get(k + '_width',
            default_widths.get(k))) for k in slice_kwargs}

        slices = {k: slice(v - slice_widths[k] / 2, v + slice_widths[k] / 2)
                  for k, v in slice_kwargs.items()}

        sliced = self._obj.sel(**slices)
        thickness = np.product([len(sliced.coords[k]) for k in slice_kwargs.keys()])
        normalized = sliced.sum(slices.keys(), keep_attrs=True) / thickness
        normalized.attrs.update(self._obj.attrs.copy())
        return normalized

    @property
    def sample_pos(self):
        return (float(self._obj.attrs['x']),
                float(self._obj.attrs['y']),
                float(self._obj.attrs['z']),)

    @property
    def chi(self):
        """
        This angle is always the puck rotation angle, i.e. the angle that spins the puck about its center while keeping
        the manipulator body fixed. Other code sometimes refers to this angle as ``phi`` or ``sample_phi``
        :return:
        """
        options = ['chi', 'Azimuth']
        for option in options:
            if option in self._obj.attrs:
                return float(self._obj.attrs[option])

        return None

    @property
    def sample_angles(self):
        return (
            self.chi,
            self.phi,
            self.polar,
            self.theta,
        )

    @property
    def theta(self):
        """
        Theta is always the manipulator angle DoF that lies along the analyzer slit.
        :return:
        """
        return float(self._obj.attrs['theta'])

    @property
    def phi(self):
        """
        Phi is always the angle along the hemisphere
        :return:
        """
        return self._obj.coords.get('phi')

    @property
    def polar(self):
        """
        Polar is always the angle perpendicular to the analyzer slit
        :return:
        """
        return float(self._obj.attrs['polar'])

    @property
    def full_coords(self):
        full_coords = {}

        full_coords.update(dict(zip(['x', 'y', 'z'], self.sample_pos)))
        full_coords.update(dict(zip(['chi', 'phi', 'polar', 'theta'], self.sample_angles)))

        full_coords.update(self._obj.coords)
        return full_coords

    @property
    def temp(self):
        warnings.warn('This is not reliable. Fill in stub for normalizing the temperature appropriately on data load.')
        return float(self._obj.attrs['TA'])

    @property
    def condensed_attrs(self):
        """
        Since we enforce camelcase on attributes, this is a reasonable filter that catches
        the ones we don't use very often.
        :return:
        """
        return {k: v for k, v in self._obj.attrs.items() if k[0].islower()}

    @property
    def referenced_scans(self):
        """
        Produces a dataframe which has all the scans which reference this one. This only makes sense for maps.
        :return:
        """

        assert(self.spectrum_type == 'map')

        df = self._obj.attrs['df']
        return df[(df.spectrum_type != 'map') & (df.ref_id == self._obj.id)]

    @property
    def fermi_surface(self):
        return self.fat_sel(eV=0)

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataarray_accessor('S')
class ARPESDataArrayAccessor(ARPESAccessorBase):
    def show(self):
        image_tool = ImageTool()
        return image_tool.make_tool(self._obj)

    def show_d2(self):
        curve_tool = CurvatureTool()
        return curve_tool.make_tool(self._obj)

    def show_band_tool(self):
        band_tool = BandTool()
        return band_tool.make_tool(self._obj)

    def fs_plot(self, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        if out is None and isinstance(out, bool):
            out = pattern.format('{}_fs'.format(self.label))
            kwargs['out'] = out
        return plotting.labeled_fermi_surface(self._obj, **kwargs)

    def cut_dispersion_plot(self, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_cut_dispersion'.format(self.label))
            kwargs['out'] = out
        return plotting.cut_dispersion_plot(self._obj, **kwargs)

    def dispersion_plot(self, pattern='{}.png',**kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_dispersion'.format(self.label))
            kwargs['out'] = out
        return plotting.fancy_dispersion(self._obj, **kwargs)

    def isosurface_plot(self, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_isosurface'.format(self.label))
            kwargs['out'] = out
        return plotting.plot_isosurface(self._obj, **kwargs)

    def _referenced_scans_for_map_plot(self, use_id=True, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        label = self._obj.attrs['id'] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_reference_scan_fs'.format(label))
            kwargs['out'] = out

        return plotting.reference_scan_fermi_surface(self._obj, **kwargs)

    def _referenced_scans_for_hv_map_plot(self, use_id=True, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        label = self._obj.attrs['id'] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_hv_reference_scan'.format(label))
            out = '{}_hv_reference_scan.png'.format(label)
            kwargs['out'] = out

        return plotting.hv_reference_scan(self._obj, **kwargs)

    def _simple_spectrum_reference_plot(self, use_id=True, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        label = self._obj.attrs['id'] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_spectrum_reference'.format(label))
            kwargs['out'] = out

        return plotting.fancy_dispersion(self._obj, **kwargs)

    def reference_plot(self, **kwargs):
        if self.spectrum_type == 'map':
            return self._referenced_scans_for_map_plot(**kwargs)
        elif self.spectrum_type == 'hv_map':
            return self._referenced_scans_for_hv_map_plot(**kwargs)
        elif self.spectrum_type == 'spectrum':
            return self._simple_spectrum_reference_plot(**kwargs)
        else:
            import pdb
            pdb.set_trace()


@xr.register_dataset_accessor('S')
class ARPESDatasetAccessor(ARPESAccessorBase):
    def polarization_plot(self, **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = '{}_spin_polarization.png'.format(self.label)
            kwargs['out'] = out
        return plotting.spin_polarized_spectrum(self._obj, **kwargs)

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