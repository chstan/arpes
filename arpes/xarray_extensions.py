import collections
import itertools
import copy
import warnings

import numpy as np
import xarray as xr

from analysis.band_analysis_utils import param_getter, param_stderr_getter
from arpes.analysis import rebin
from typing import Optional, Union
from arpes.typing import DataType

from scipy import ndimage as ndi
from skimage import feature

import arpes.constants
import arpes.materials
from arpes.models.band import MultifitBand
from arpes.io import load_dataset_attrs
import arpes.plotting as plotting
from arpes.plotting import ImageTool, CurvatureTool, BandTool, FitCheckTool
from arpes.utilities.conversion import slice_along_path
from arpes.utilities import apply_dataarray
from arpes.utilities.region import DesignatedRegions, normalize_region
from utilities.math import shift_by

__all__ = ['ARPESDataArrayAccessor', 'ARPESDatasetAccessor', 'ARPESFitToolsAccessor']


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
    def is_subtracted(self):
        if self._obj.attrs.get('subtracted'):
            return True

        if isinstance(self._obj, xr.DataArray):
            # if at least 5% of the values are < 0 we should consider the data
            # to be best represented by a coolwarm map
            return (((self._obj < 0) * 1).mean() > 0.05).item()

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
        try:
            history = self.history
            if len(history) >= 3:
                first_modification = history[-3]
                df = self._obj.attrs['df']
                return df[df.id == first_modification['parent_id']].index[0]
        except:
            pass
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
        except (IndexError, KeyError, AttributeError):
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

    def find_spectrum_energy_edges(self, indices=False):
        energy_marginal = self._obj.sum([d for d in self._obj.dims if d not in ['eV']])

        embed_size = 20
        embedded = np.ndarray(shape=[embed_size] + list(energy_marginal.values.shape))
        embedded[:] = energy_marginal.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        edges = feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True,
                              low_threshold=0.2) * 1
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.T.stride(generic_dim_names=False)
        return edges * delta['eV'] + self._obj.coords['eV'].values[0]

    def find_spectrum_angular_edges_full(self, indices=False):
        # as a first pass, we need to find the bottom of the spectrum, we will use this
        # to select the active region and then to rebin into course steps in energy from 0
        # down to this region
        # we will then find the appropriate edge for each slice, and do a fit to the edge locations

        energy_edge = self.find_spectrum_energy_edges()
        low_edge = np.min(energy_edge) + 0.05
        high_edge = np.max(energy_edge) - 0.05

        if high_edge - low_edge < 0.15:
            # Doesn't look like the automatic inference of the energy edge was valid
            high_edge = 0
            low_edge = np.min(self._obj.coords['eV'].values)

        angular_dim = 'pixel' if 'pixel' in self._obj.dims else 'phi'
        energy_cut = self._obj.sel(eV=slice(low_edge, high_edge)).S.sum_other(['eV', angular_dim])

        n_cuts = int(np.ceil(high_edge - low_edge / 0.05))
        new_shape = {'eV': n_cuts}
        new_shape[angular_dim] = len(energy_cut.coords[angular_dim].values)
        rebinned = rebin(energy_cut, shape=new_shape)

        embed_size = 20
        embedded = np.ndarray(shape=[embed_size] + [len(rebinned.coords[angular_dim].values)])
        low_edges = []
        high_edges = []
        for e_cut in rebinned.coords['eV'].values:
            e_slice = rebinned.sel(eV=e_cut)
            values = e_slice.values
            values[values > np.mean(values)] = np.mean(values)
            embedded[:] = values
            embedded = ndi.gaussian_filter(embedded, embed_size / 1.5)

            edges = feature.canny(embedded, sigma=4, use_quantiles=False,
                                  low_threshold=0.7, high_threshold=1.5) * 1
            edges = np.where(edges[int(embed_size / 2)] == 1)[0]
            low_edges.append(np.min(edges))
            high_edges.append(np.max(edges))

        if indices:
            return np.array(low_edges), np.array(high_edges), rebinned.coords['eV']

        delta = self._obj.T.stride(generic_dim_names=False)

        low_edges = np.array(low_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0]
        high_edges = np.array(high_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0]

        # DEBUG
        # import matplotlib.pyplot as plt
        # energy_cut.plot()
        # plt.plot(rebinned.coords['eV'], low_edges)
        # plt.plot(rebinned.coords['eV'], high_edges)

        return low_edges, high_edges, rebinned.coords['eV']


    def zero_spectrometer_edges(self, edge_type='hard', cut_margin=None, interp_range=None, low=None, high=None):
        """
        At the moment we only provide hard edges. Soft edges would be smoothed
        with a logistic function or similar.
        :param edge_type:
        :param inter_range: Range over which to extrapolate fit
        :return:
        """

        if low is not None:
            assert(high is not None)
            assert(len(low) == len(high) == 2)

            low_edges = low
            high_edges = high

        low_edges, high_edges, rebinned_eV_coord = self.find_spectrum_angular_edges_full(indices=True)




        angular_dim = 'pixel' if 'pixel' in self._obj.dims else 'phi'
        if cut_margin is None:
            if 'pixel' in self._obj.dims:
                cut_margin = 50
            else:
                cut_margin = int(0.08 / self._obj.T.stride(generic_dim_names=False)[angular_dim])
        else:
            if isinstance(cut_margin, float):
                assert(angular_dim == 'phi')
                cut_margin = int(cut_margin / self._obj.T.stride(generic_dim_names=False)[angular_dim])

        if interp_range is not None:
            low_edge = xr.DataArray(low_edges, coords={'eV': rebinned_eV_coord}, dims=['eV'])
            high_edge = xr.DataArray(high_edges, coords={'eV': rebinned_eV_coord}, dims=['eV'])
            low_edge = low_edge.sel(eV=interp_range)
            high_edge = high_edge.sel(eV=interp_range)
            import pdb
            pdb.set_trace()

        other_dims = list(self._obj.dims)
        other_dims.remove('eV')
        other_dims.remove(angular_dim)
        copied = self._obj.copy(deep=True).transpose(*(['eV', angular_dim] + other_dims))

        low_edges += cut_margin
        high_edges -= cut_margin

        for i, energy in enumerate(copied.coords['eV'].values):
            index = np.searchsorted(rebinned_eV_coord, energy)
            other = index + 1
            if other >= len(rebinned_eV_coord):
                other = len(rebinned_eV_coord) - 1
                index = len(rebinned_eV_coord) - 2

            low = int(np.interp(energy, rebinned_eV_coord, low_edges))
            high = int(np.interp(energy, rebinned_eV_coord, high_edges))
            copied.values[i, 0:low] = 0
            copied.values[i, high:-1] = 0

        return copied

    def sum_other(self, dim_or_dims):
        if isinstance(dim_or_dims, str):
            dim_or_dims = [dim_or_dims]

        return self._obj.sum([d for d in self._obj.dims if d not in dim_or_dims])

    def find_spectrum_angular_edges(self, indices=False):
        angular_dim = 'pixel' if 'pixel' in self._obj.dims else 'phi'
        energy_edge = self.find_spectrum_energy_edges()
        energy_slice = slice(np.max(energy_edge) - 0.1, np.max(energy_edge))
        near_ef = self._obj.sel(eV=energy_slice).sum([d for d in self._obj.dims if d not in [angular_dim]])

        embed_size = 20
        embedded = np.ndarray(shape=[embed_size] + list(near_ef.values.shape))
        embedded[:] = near_ef.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        edges = feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True,
                              low_threshold=0.2) * 1
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.T.stride(generic_dim_names=False)
        return edges * delta[angular_dim] + self._obj.coords[angular_dim].values[0]

    def trimmed_selector(self):
        raise NotImplementedError()

    def wide_angle_selector(self, include_margin=True):
        edges = self.find_spectrum_angular_edges()
        low_edge, high_edge = np.min(edges), np.max(edges)

        # go and build in a small margin
        if include_margin:
            if 'pixels' in self._obj.dims:
                low_edge += 50
                high_edge -= 50
            else:
                low_edge += 0.05
                high_edge -= 0.05

        return slice(low_edge, high_edge)

    def narrow_angle_selector(self):
        raise NotImplementedError()

    def meso_effective_selector(self):
        energy_edge = self.find_spectrum_energy_edges()
        return slice(np.max(energy_edge) - 0.3, np.max(energy_edge) - 0.1)

    def region_sel(self, *regions):
        def process_region_selector(selector: Union[slice, DesignatedRegions], dimension_name: str):
            if isinstance(selector, slice):
                return selector

            # need to read out the region
            options = {
                'eV': (DesignatedRegions.ABOVE_EF,
                       DesignatedRegions.BELOW_EF,
                       DesignatedRegions.EF_NARROW,
                       DesignatedRegions.MESO_EF,

                       DesignatedRegions.MESO_EFFECTIVE_EF,
                       DesignatedRegions.ABOVE_EFFECTIVE_EF,
                       DesignatedRegions.BELOW_EFFECTIVE_EF,
                       DesignatedRegions.EFFECTIVE_EF_NARROW),

                'phi': (DesignatedRegions.NARROW_ANGLE,
                        DesignatedRegions.WIDE_ANGLE,
                        DesignatedRegions.TRIM_EMPTY),
            }

            options_for_dim = options.get(dimension_name, [d for d in DesignatedRegions])
            assert(selector in options_for_dim)

            # now we need to resolve out the region
            resolution_methods = {
                DesignatedRegions.ABOVE_EF: slice(0, None),
                DesignatedRegions.BELOW_EF: slice(None, 0),
                DesignatedRegions.EF_NARROW: slice(-0.1, 0.1),  # TODO do this better
                DesignatedRegions.MESO_EF: slice(-0.3, -0.1),
                DesignatedRegions.MESO_EFFECTIVE_EF: self.meso_effective_selector,

                # Implement me
                #DesignatedRegions.TRIM_EMPTY: ,
                DesignatedRegions.WIDE_ANGLE: self.wide_angle_selector,
                #DesignatedRegions.NARROW_ANGLE: self.narrow_angle_selector,
            }
            resolution_method = resolution_methods[selector]
            if isinstance(resolution_method, slice):
                return resolution_method
            elif callable(resolution_method):
                return resolution_method()
            else:
                raise NotImplementedError('FIXME')

        obj = self._obj

        def unpack_dim(dim_name):
            if dim_name == 'angular':
                return 'pixel' if 'pixel' in obj.dims else 'phi'

            return dim_name

        for region in regions:
            region = {unpack_dim(k): v for k, v in normalize_region(region).items()}

            # remove missing dimensions from selection for permissiveness
            # and to transparent composing of regions
            region = {k: process_region_selector(v, k)
                      for k, v in region.items() if k in obj.dims}
            obj = obj.sel(**region)

        return obj

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
            'eV': 0.05,
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
    def spectrometer_settings(self):
        find_keys = {
            'lens_mode',
            'pass_energy',
            'scan_mode',
            'scan_region',
            'slit',
        }

        return {k: v for k, v in self._obj.attrs.items() if k in find_keys}

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


    def subtraction_reference_plot(self, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_subtraction_reference'.format(self.label))
            kwargs['out'] = out

        return plotting.tarpes.plot_subtraction_reference(self._obj, **kwargs)

    def fermi_edge_reference_plot(self, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_fermi_edge_reference'.format(self.label))
            kwargs['out'] = out

        return plotting.fermi_edge.fermi_edge_reference(self._obj, **kwargs)

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


NORMALIZED_DIM_NAMES = ['x', 'y', 'z', 'w']

@xr.register_dataarray_accessor('T')
class GenericAccessorTools(object):
    _obj = None

    def clean_outliers(self, clip=0.5):
        low, high = np.percentile(self._obj.values, [clip, 100 - clip])
        copy = self._obj.copy(deep=True)
        copy.values[copy.values < low] = low
        copy.values[copy.values > high] = high
        return copy

    def as_movie(self, time_dim=None, pattern='{}.png', **kwargs):
        if time_dim is None:
            time_dim = self._obj.dims[-1]

        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_animation'.format(self.label))
            kwargs['out'] = out
        return plotting.plot_movie(self._obj, time_dim, **kwargs)


    def filter_coord(self, coordinate_name, sieve):
        """
        Filters a dataset along a coordinate.

        Sieve should be a function which accepts a coordinate value and the slice
        of the data along that dimension.

        :param coordinate_name:
        :param sieve:
        :return:
        """
        mask = np.array([i for i, c in enumerate(self._obj.coords[coordinate_name])
                         if sieve(c, self._obj.isel(**dict([[coordinate_name, i]])))])
        return self._obj.isel(**dict([[coordinate_name, mask]]))

    def iterate_axis(self, axis_name_or_axes):
        if isinstance(axis_name_or_axes, int):
            axis_name_or_axes = self._obj.dims[axis_name_or_axes]

        if isinstance(axis_name_or_axes, str):
            axis_name_or_axes = [axis_name_or_axes]

        coord_iterators = [self._obj.coords[d].values for d in axis_name_or_axes]
        for indices in itertools.product(*[range(len(c)) for c in coord_iterators]):
            cut_coords = [cs[index] for cs, index in zip(coord_iterators, indices)]
            coords_dict = dict(zip(axis_name_or_axes, cut_coords))
            yield coords_dict, self._obj.sel(method='nearest', **coords_dict)


    def map(self, fn):
        return apply_dataarray(self._obj, np.vectorize(fn))

    def enumerate_iter_coords(self):
        coords_list = [self._obj.coords[d].values for d in self._obj.dims]
        for indices in itertools.product(*[range(len(c)) for c in coords_list]):
            cut_coords = [cs[index] for cs, index in zip(coords_list, indices)]
            yield indices, dict(zip(self._obj.dims, cut_coords))

    def iter_coords(self, dim_names=None):
        if dim_names is None:
            dim_names = self._obj.dims
        for ts in itertools.product(*[self._obj.coords[d].values for d in self._obj.dims]):
            yield dict(zip(self._obj.dims, ts))

    def range(self, generic_dim_names=True):
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_ranges = [(np.min(coord.values), np.max(coord.values)) for coord in indexed_coords]

        dim_names = self._obj.dims
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[:len(dim_names)]

        return dict(zip(dim_names, indexed_ranges))

    def stride(self, generic_dim_names=True):
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_strides = [coord.values[1] - coord.values[0] for coord in indexed_coords]

        dim_names = self._obj.dims
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[:len(dim_names)]

        return dict(zip(dim_names, indexed_strides))

    def shift_by(self, other, shift_axis=None, zero_nans=True, shift_coords=False):
        # for now we only support shifting by a one dimensional array

        data = self._obj
        assert(len(other.dims) == 1)

        if shift_coords:
            mean_shift = np.mean(other)
            other -= mean_shift

        by_axis = other.dims[0]
        if shift_axis is None:
            option_dims = list(data.dims)
            option_dims.remove(by_axis)
            assert(len(option_dims) == 1)
            shift_axis = option_dims[0]

        shift_amount = -other.values / data.T.stride(generic_dim_names=False)[shift_axis]

        shifted_data = shift_by(data.values, shift_amount,
                     axis=list(data.dims).index(shift_axis),
                     by_axis=list(data.dims).index(by_axis), order=1)

        if zero_nans:
            shifted_data[np.isnan(shifted_data)] = 0

        coords = copy.deepcopy(data.coords)
        if shift_coords:
            coords[shift_axis] -= mean_shift

        return xr.DataArray(
            shifted_data,
            coords,
            data.dims,
            attrs=data.attrs.copy(),
        )


    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj


@xr.register_dataset_accessor('F')
class ARPESDatasetFitToolAccessor(object):
    _obj = None

    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj

    def show(self):
        fit_diagnostic_tool = FitCheckTool()
        return fit_diagnostic_tool.make_tool(self._obj)

    def p(self, param_name):
        return self._obj.results.F.p(param_name)

    def s(self, param_name):
        return self._obj.results.F.s(param_name)

@xr.register_dataarray_accessor('F')
class ARPESFitToolsAccessor(object):
    _obj = None

    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj

    def show(self):
        fit_diagnostic_tool = FitCheckTool()
        return fit_diagnostic_tool.make_tool(self._obj)

    def show_fit_diagnostic(self):
        return self.show()

    def p(self, param_name):
        return self._obj.T.map(param_getter(param_name))

    def s(self, param_name):
        return self._obj.T.map(param_stderr_getter(param_name))

    @property
    def bands(self):
        """
        This should probably instantiate appropriate types
        :return:
        """
        band_names = self.band_names

        bands = {l: MultifitBand(label=l, data=self._obj) for l in band_names}

        return bands

    @property
    def band_names(self):
        collected_band_names = set()

        for item in self._obj.values.ravel():
            if item is None:
                continue

            band_names = [k[:-6] for k in item.params.keys() if 'center' in k]
            collected_band_names = collected_band_names.union(set(band_names))

        return collected_band_names

    @property
    def parameter_names(self):
        collected_parameter_names = set()

        for item in self._obj.values.ravel():
            if item is None:
                continue

            param_names = [k for k in item.params.keys()]
            collected_parameter_names = collected_parameter_names.union(set(param_names))

        return collected_parameter_names


    def show_fit_diagnostic(self):
        """
        alias for ``.show``
        :return:
        """
        return self.show()


@xr.register_dataset_accessor('S')
class ARPESDatasetAccessor(ARPESAccessorBase):
    def __getattr__(self, item):
        return getattr(self._obj.S.spectrum.S, item)

    def polarization_plot(self, **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
            out = '{}_spin_polarization.png'.format(self.label)
            kwargs['out'] = out
        return plotting.spin_polarized_spectrum(self._obj, **kwargs)

    @property
    def spectrum(self) -> Optional[xr.DataArray]:
        spectrum = None
        if 'spectrum' in self._obj.data_vars:
            spectrum = self._obj.spectrum
        elif 'raw' in self._obj.data_vars:
            spectrum = self._obj.raw
        elif '__xarray_dataarray_variable__' in self._obj.data_vars:
            spectrum = self._obj.__xarray_dataarray_variable__

        if spectrum is not None and 'df' not in spectrum.attrs:
            spectrum.attrs['df'] = self._obj.attrs.get('df', None)

        return spectrum

    @property
    def spectrum_type(self):
        try:
            # this isn't the smartest thing in the world,
            # but it should allow some old code to keep working on datasets transparently
            return self.spectrum.S.spectrum_type
        except Exception:
            return 'dataset'

    @property
    def degrees_of_freedom(self):
        return set(self.spectrum.dims)

    @property
    def spectrum_degrees_of_freedom(self):
        return self.degrees_of_freedom.intersection({
            'eV', 'phi', 'pixel', 'kx', 'kp'
        })

    @property
    def scan_degrees_of_freedom(self):
        return self.degrees_of_freedom.difference(self.spectrum_degrees_of_freedom)

    def reference_plot(self, **kwargs):
        """
        A bit of a misnomer because this actually makes many plots. For full datasets, the relevant components
        are:

        1. Temperature as function of scan DOF
        2. Photocurrent as a function of scan DOF
        3. Photocurrent normalized + unnormalized figures, in particular
           i. The reference plots for the photocurrent normalized spectrum
           ii. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF integrated over E, phi
           iii. For delay scans:
             1. Fermi location as a function of scan DoF, integrated over phi
             2. Subtraction scans


        :param kwargs:
        :return:
        """
        scan_dofs_integrated = self._obj.sum(*list(self.scan_degrees_of_freedom))
        original_out = kwargs.get('out')

        # make figures for temperature, photocurrent, delay
        make_figures_for = ['T', 'IG_nA', 'current', 'photocurrent']
        name_normalization = {
            'T': 'T',
            'IG_nA': 'photocurrent',
            'current': 'photocurrent'
        }

        for figure_item in make_figures_for:
            if figure_item not in self._obj.data_vars:
                continue

            name = name_normalization.get(figure_item, figure_item)
            data_var = self._obj[figure_item]
            out = '{}_{}_spec_integrated_reference.png'.format(self.label, name)
            return plotting.scan_var_reference_plot(
                data_var, title='Reference {}'.format(name), out=out)


        # may also want to make reference figures summing over cycle, or summing over beta


        # make photocurrent normalized figures
        try:
            normalized = self._obj / self._obj.IG_nA
            normalized.S.make_spectrum_reference_plots(prefix='norm_PC_', out=True)
        except:
            pass

        self.make_spectrum_reference_plots(out=True)

    def make_spectrum_reference_plots(self, prefix='', **kwargs):
        """
        Photocurrent normalized + unnormalized figures, in particular:

        i. The reference plots for the photocurrent normalized spectrum
        ii. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF integrated over E, phi
        iii. For delay scans:
          1. Fermi location as a function of scan DoF, integrated over phi
          2. Subtraction scans
        """
        self.spectrum.S.reference_plot(pattern=prefix + '{}.png', **kwargs)
        if 'cycle' in self._obj.coords:
            integrated_over_scan = self._obj.sum(*list(self.spectrum_degrees_of_freedom))
            integrated_over_scan.S.spectrum.S.reference_plot(pattern=prefix + 'sum_spec_DoF_{}.png', **kwargs)

        if 'delay' in self._obj.coords:
            # TODO fermi location scans
            dims = self.spectrum_degrees_of_freedom
            dims.remove('eV')
            angle_integrated = self._obj.sum(*list(dims))

            # subtraction scan
            self.spectrum.S.subtraction_reference_plots(pattern=prefix + '{}.png', **kwargs)
            angle_integrated.S.fermi_edge_reference_plots(pattern=prefix + '{}.png', **kwargs)

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
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