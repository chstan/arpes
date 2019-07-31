import collections
import itertools
import copy
import warnings
from collections import OrderedDict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from arpes.plotting.utils import fancy_labels, remove_colorbars
from typing import Optional, Union

from arpes.analysis.band_analysis_utils import param_getter, param_stderr_getter
from arpes.analysis import rebin
from arpes.exceptions import AnalysisError
from arpes.typing import DataType

from scipy import ndimage as ndi

import arpes.constants
import arpes.materials
from arpes.models.band import MultifitBand
from arpes.io import load_dataset_attrs
import arpes.plotting as plotting
from arpes.plotting import ImageTool, CurvatureTool, BandTool, FitCheckTool
from arpes.utilities.conversion import slice_along_path
from arpes.utilities import apply_dataarray
from arpes.utilities.region import DesignatedRegions, normalize_region
from arpes.utilities.math import shift_by

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

    def find(self, name):
        return [n for n in dir(self) if name in n]

    @property
    def sherman_function(self):
        return 0.2

        for option in ['sherman', 'sherman_function', 'SHERMAN']:
            if option in self._obj.attrs:
                return self._obj.attrs[option]

        return 0.2

    @property
    def experimental_conditions(self):
        try:
            temp = self.temp
        except AttributeError:
            temp = None

        return {
            'hv': self.hv,
            'polarization': self.polarization,
            'temp': temp,
        }

    @property
    def polarization(self):
        if 'epu_pol' in self._obj.attrs:
            # merlin: TODO normalize these
            # check and complete
            try:
                return {
                    0: 'p',
                    1: 'rc',
                    2: 's',
                }.get(int(self._obj.attrs['epu_pol']))
            except ValueError:
                return self._obj.attrs['epu_pol']

        return None

    @property
    def is_subtracted(self):
        if self._obj.attrs.get('subtracted'):
            return True

        if isinstance(self._obj, xr.DataArray):
            # if at least 5% of the values are < 0 we should consider the data
            # to be best represented by a coolwarm map
            return (((self._obj < 0) * 1).mean() > 0.05).item()

    @property
    def is_spatial(self):
        """
        Infers whether a given scan has real-space dimensions and corresponds to
        SPEM or u/nARPES.
        :return:
        """
        if self.spectrum_type in {'ucut', 'spem'}:
            return True

        return any(d in {'X', 'Y', 'Z'} for d in self._obj.dims)

    @property
    def is_kspace(self):
        """
        Infers whether the scan is k-space converted or not. Because of the way this is defined, it will return
        true for XPS spectra, which I suppose is true but trivially.
        :return:
        """
        return not any(d in {'phi', 'theta', 'beta', 'angle'} for d in self._obj.dims)

    @property
    def is_slit_vertical(self):
        spectrometer = self.spectrometer
        if spectrometer is not None:
            try:
                return spectrometer['is_slit_vertical']
            except KeyError:
                pass

        if 'is_slit_vertical' in self._obj.attrs:
            return self._obj.attrs['is_slit_vertical']

        raise AnalysisError('Unknown spectrometer configuration.')

    @property
    def endstation(self):
        return self._obj.attrs['location']

    @property
    def is_synchrotron(self):
        endstation = self.endstation

        synchrotron_endstations = {
            'BL403',
            'BL702',
        }

        return endstation in synchrotron_endstations

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
        if 'spectrum_type' in self._obj.attrs and self._obj.attrs['spectrum_type']:
            return self._obj.attrs['spectrum_type']

        dim_types = {
            ('eV',): 'xps_spectrum',
            ('eV', 'phi'): 'spectrum',

            # this should check whether the other angular axis perpendicular to scan axis?
            ('eV', 'phi', 'beta'): 'map',
            ('eV', 'phi', 'theta'): 'map',

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

    def transpose_to_front(self, dim):
        dims = list(self._obj.dims)
        assert(dim in dims)
        dims.remove(dim)
        return self._obj.transpose(*([dim] + dims))

    def transpose_to_back(self, dim):
        dims = list(self._obj.dims)
        assert(dim in dims)
        dims.remove(dim)
        return self._obj.transpose(*(dims + [dim]))

    def select_around_data(self, points, radius=None, fast=False, safe=True, mode='sum', **kwargs):
        """
        Can be used to perform a selection along one axis as a function of another, integrating a region
        in the other dimensions. As an example, suppose we have a dataset with dimensions ('eV', 'kp', 'T',)
        and we also by fitting determined the Fermi momentum as a function of T, kp_F('T'), stored in the
        dataarray kFs. Then we could select momentum integrated EDCs in a small window around the fermi momentum
        for each temperature by using

        ```python
        edcs_at_fermi_momentum = full_data.S.select_around_data({'kp': kFs}, radius={'kp': 0.04}, fast=True)
        ```

        The resulting data will be EDCs for each T, in a region of radius 0.04 inverse angstroms around the
        Fermi momentum.

        :param points:
        :param radius:
        :param fast:
        :param safe:
        :param kwargs:
        :return:
        """
        if isinstance(self._obj, xr.Dataset):
            raise TypeError('Cannot use select_around on Datasets only DataArrays!')

        if mode not in {'sum', 'mean'}:
            raise ValueError('mode parameter should be either sum or mean.')

        if isinstance(points, (tuple, list,)):
            warnings.warn('Dangerous iterable points argument to `select_around`')
            points = dict(zip(points, self._obj.dims))
        if isinstance(points, xr.Dataset):
            points = {k: points[k].item() for k in points.data_vars}

        default_radii = {
            'kp': 0.02,
            'kz': 0.05,
            'phi': 0.02,
            'beta': 0.02,
            'theta': 0.02,
            'eV': 0.05,
            'delay': 0.2,
            'T': 2,
            'temp': 2,
        }

        unspecified = 0.1

        if isinstance(radius, float):
            radius = {d: radius for d in points.keys()}
        else:
            collected_terms = set('{}_r'.format(k) for k in points.keys()).intersection(
                set(kwargs.keys()))
            if len(collected_terms):
                radius = {d: kwargs.get('{}_r'.format(d), default_radii.get(d, unspecified))
                          for d in points.keys()}
            elif radius is None:
                radius = {d: default_radii.get(d, unspecified) for d in points.keys()}

        assert (isinstance(radius, dict))
        radius = {d: radius.get(d, default_radii.get(d, unspecified)) for d in points.keys()}

        along_dims = list(points.values())[0].dims
        selected_dims = list(points.keys())

        stride = self._obj.T.stride(generic_dim_names=False)

        new_dim_order = [d for d in self._obj.dims if d not in along_dims] + list(along_dims)

        data_for = self._obj.transpose(*new_dim_order)
        new_data = data_for.sum(selected_dims, keep_attrs=True)
        for coord, value in data_for.T.iterate_axis(along_dims):
            nearest_sel_params = {}
            if safe:
                for d, v in radius.items():
                    if v < stride[d]:
                        nearest_sel_params[d] = points[d].sel(**coord)

                radius = {d: v for d, v in radius.items() if d not in nearest_sel_params}

            if fast:
                selection_slices = {d: slice(points[d].sel(**coord) - radius[d], points[d].sel(**coord) + radius[d])
                                    for d in points.keys() if d in radius}
                selected = value.sel(**selection_slices)
            else:
                raise NotImplementedError()

            if len(nearest_sel_params):
                selected = selected.sel(**nearest_sel_params, method='nearest')

            for d in nearest_sel_params:
                # need to remove the extra dims from coords
                del selected.coords[d]

            if mode == 'sum':
                new_data.loc[coord] = selected.sum(list(radius.keys())).values
            elif mode == 'mean':
                new_data.loc[coord] = selected.mean(list(radius.keys())).values

        return new_data


    def select_around(self, point, radius=None, fast=False, safe=True, mode='sum', **kwargs):
        """
        Selects and integrates a region around a one dimensional point, useful to do a small
        region integration, especially around points on a path of a k-point of interest.

        If the fast flag is set, we will use the Manhattan norm, i.e. sum over square regions
        rather than ellipsoids, as this is less costly.

        If radii are not set, or provided through kwargs as 'eV_r' or 'phi_r' for instance,
        then we will try to use reasonable default values; buyer beware.
        :param point:
        :param radius:
        :param fast:
        :return:
        """
        if isinstance(self._obj, xr.Dataset):
            raise TypeError('Cannot use select_around on Datasets only DataArrays!')

        if mode not in {'sum', 'mean'}:
            raise ValueError('mode parameter should be either sum or mean.')

        if isinstance(point, (tuple, list,)):
            warnings.warn('Dangerous iterable point argument to `select_around`')
            point = dict(zip(point, self._obj.dims))
        if isinstance(point, xr.Dataset):
            point = {k: point[k].item() for k in point.data_vars}

        default_radii = {
            'kp': 0.02,
            'kz': 0.05,
            'phi': 0.02,
            'beta': 0.02,
            'theta': 0.02,
            'eV': 0.05,
            'delay': 0.2,
            'T': 2,
        }

        unspecified = 0.1

        if isinstance(radius, float):
            radius = {d: radius for d in point.keys()}
        else:
            collected_terms = set('{}_r'.format(k) for k in point.keys()).intersection(
                set(kwargs.keys()))
            if len(collected_terms):
                radius = {d: kwargs.get('{}_r'.format(d), default_radii.get(d, unspecified))
                          for d in point.keys()}
            elif radius is None:
                radius = {d: default_radii.get(d, unspecified) for d in point.keys()}

        assert (isinstance(radius, dict))
        radius = {d: radius.get(d, default_radii.get(d, unspecified)) for d in point.keys()}

        # make sure we are taking at least one pixel along each
        nearest_sel_params = {}
        if safe:
            stride = self._obj.T.stride(generic_dim_names=False)
            for d, v in radius.items():
                if v < stride[d]:
                    nearest_sel_params[d] = point[d]

            radius = {d: v for d, v in radius.items() if d not in nearest_sel_params}

        if fast:
            selection_slices = {d: slice(point[d] - radius[d], point[d] + radius[d])
                                for d in point.keys() if d in radius}
            selected = self._obj.sel(**selection_slices)
        else:
            # selected = self._obj
            raise NotImplementedError()

        if len(nearest_sel_params):
            selected = selected.sel(**nearest_sel_params, method='nearest')

        for d in nearest_sel_params:
            # need to remove the extra dims from coords
            del selected.coords[d]

        if mode == 'sum':
            return selected.sum(list(radius.keys()))
        elif mode == 'mean':
            return selected.mean(list(radius.keys()))

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
            'BL7': arpes.constants.SPECTROMETER_BL7,
            'ANTARES': arpes.constants.SPECTROMETER_ANTARES,
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
                'Kaindl': arpes.constants.SPECTROMETER_KAINDL,
                'BL7': arpes.constants.SPECTROMETER_BL7,
                'ANTARES': arpes.constants.SPECTROMETER_ANTARES,
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
    def scan_row(self):
        df = self._obj.attrs['df']
        sdf = df[df.path == self._obj.attrs['file']]
        return list(sdf.iterrows())[0]

    @property
    def df_index(self):
        return self.scan_row[0]

    @property
    def df_after(self):
        return self._obj.attrs['df'][self._obj.attrs['df'].index > self.df_index]

    def df_until_type(self, df=None, spectrum_type=None):
        if df is None:
            df = self.df_after

        if spectrum_type is None:
            spectrum_type = (self.spectrum_type,)

        if isinstance(spectrum_type, str):
            spectrum_type = (spectrum_type,)

        try:
            indices = [df[df['spectrum_type'].eq(s)] for s in spectrum_type]
            indices = [d.index[0] for d in indices if not d.empty]

            if len(indices) == 0:
                raise IndexError()

            min_index = min(indices)
            return df[df.index < min_index]
        except IndexError:
            # nothing
            return df

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
        return str(self._obj.attrs.get('description', self.scan_name))

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

    def lookup_offset_coord(self, name):
        offset = self.lookup_coord(name) - self._lookup_offset(name)
        try:
            return offset.item()
        except (AttributeError, ValueError):
            try:
                return offset.values
            except AttributeError:
                return offset

    def lookup_coord(self, name):
        if name in self._obj.coords:
            return self._obj.coords[name]

        if name in self._obj.attrs:
            return self._obj.attrs[name]

        raise ValueError('Could not find coordinate {}.'.format(name))

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
    def beta_offset(self):
        return self._lookup_offset('beta')

    @property
    def theta_offset(self):
        return self._lookup_offset('theta')

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

        from skimage import feature  # try to avoid dependency conflict with numpy v0.16
        edges = feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True,
                              low_threshold=0.3) * 1
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

            from skimage import feature  # try to avoid dependency conflict with numpy v0.16
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

        from skimage import feature  # try to avoid dependency conflict with numpy v0.16
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
            'beta': 2,
            'theta': 2,
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
        for k, v in slices.items():
            normalized.coords[k] = (v.start + v.stop) / 2
        normalized.attrs.update(self._obj.attrs.copy())
        return normalized

    @property
    def reference_settings(self):
        settings = self.spectrometer_settings or {}

        settings.update({
            'hv': self.hv,
        })

        return settings

    @property
    def beamline_settings(self):
        find_keys = {
            'entrance_slit': {'entrance_slit',},
            'exit_slit': {'exit_slit',},
            'hv': {'hv', 'photon_energy',},
            'grating': {},
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        if self.endstation == 'BL403':
            settings['grating'] = 'HEG' # for now assume we always use the first order light

        return settings

    @property
    def spectrometer_settings(self):
        find_keys = {
            'lens_mode': {'lens_mode',},
            'pass_energy': {'pass_energy', },
            'scan_mode': {'scan_mode',},
            'scan_region': {'scan_region',},
            'slit': {'slit', 'slit_plate',},
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        if isinstance(settings.get('slit'), (float, np.float32, np.float64)):
            settings['slit'] = int(round(settings['slit']))

        return settings

    @property
    def sample_pos(self):
        x, y, z = None, None, None
        try:
            x = self._obj.attrs['x']
        except KeyError:
            pass
        try:
            y = self._obj.attrs['y']
        except KeyError:
            pass
        try:
            z = self._obj.attrs['z']
        except KeyError:
            pass

        def do_float(w):
            return float(w) if w is not None else None

        return (do_float(x), do_float(y), do_float(z))

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
            # manipulator
            self.beta,
            self.theta,
            self.chi,

            # analyzer
            self.phi,
            self.psi,
            self.alpha,
        )

    """
    For a description of the PyARPES angular conventions, visit
    """
    @property
    def theta(self):
        try:
            return float(self._obj.attrs['theta'])
        except KeyError:
            return None

    @property
    def phi(self):
        try:
            return float(self._obj.attrs['phi'])
        except KeyError:
            return None

    @property
    def beta(self):
        try:
            return float(self._obj.attrs['beta'])
        except KeyError:
            return None

    @property
    def full_coords(self):
        full_coords = {}

        full_coords.update(dict(zip(['x', 'y', 'z'], self.sample_pos)))
        full_coords.update(dict(zip(['beta', 'theta', 'chi', 'phi', 'psi', 'alpha'], self.sample_angles)))
        full_coords.update({
            'hv': self.hv,
        })

        full_coords.update(self._obj.coords)
        return full_coords

    @property
    def temp(self):
        """
        TODO, agressively normalize attributes across different spectrometers
        :return:
        """
        prefered_attrs = ['TA', 'ta', 't_a', 'T_A', 'T_1', 't_1', 't1', 'T1', 'temp', 'temp_sample', 'temp_cryotip',
                          'temperature_sensor_b', 'temperature_sensor_a']
        for attr in prefered_attrs:
            if attr in self._obj.attrs:
                return float(self._obj.attrs[attr])

        raise AttributeError('Could not read temperature off any standard attr')

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

        if self.spectrum_type == 'map':
            df = self._obj.attrs['df']
            return df[(df.spectrum_type != 'map') & (df.ref_id == self._obj.id)]
        else:
            assert(self.spectrum_type in {'ucut', 'spem'})
            df = self.df_until_type(spectrum_type=('ucut', 'spem',))
            return df

    def generic_fermi_surface(self, fermi_energy):
        return self.fat_sel(eV=fermi_energy)

    @property
    def fermi_surface(self):
        return self.fat_sel(eV=0)

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


    def dict_to_html(self, d):
        return """
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(
            rows=''.join(['<tr><td>{}</td><td>{}</td></tr>'.format(k, v) for k, v in d.items()])
        )

    def _repr_html_full_coords(self, coords):
        def coordinate_dataarray_to_flat_rep(value):
            if not isinstance(value, xr.DataArray):
                return value

            return '<span>{min:.3g}<strong> to </strong>{max:.3g}<strong> by </strong>{delta:.3g}</span>'.format(
                min=value.min().item(),
                max=value.max().item(),
                delta=value.values[1] - value.values[0],
            )

        return self.dict_to_html({k: coordinate_dataarray_to_flat_rep(v) for k, v in coords.items()})

    def _repr_html_spectrometer_info(self):
        skip_keys = {'dof',}
        ordered_settings = OrderedDict(self.spectrometer_settings)
        ordered_settings.update({k: v for k, v in self.spectrometer.items()
                                 if k not in skip_keys})

        return self.dict_to_html(ordered_settings)

    def _repr_html_experimental_conditions(self, conditions):
        transforms = {
            'polarization': lambda p: {
                'p': 'Linear Horizontal',
                's': 'Linear Vertical',
                'rc': 'Right Circular',
                'lc': 'Left Circular',
                's-p': 'Linear Dichroism',
                'p-s': 'Linear Dichroism',
                'rc-lc': 'Circular Dichroism',
                'lc-rc': 'Circular Dichroism',
            }.get(p, p),
            'hv': lambda hv: '{} eV'.format(hv),
            'temp': lambda temp: '{} Kelvin'.format(temp),
        }

        id = lambda x: x

        return self.dict_to_html({k: transforms.get(k, id)(v) for k, v in conditions.items() if v is not None})

    def _repr_html_(self):
        skip_data_vars = {'time',}

        if isinstance(self._obj, xr.Dataset):
            to_plot = [k for k in self._obj.data_vars.keys() if k not in skip_data_vars]
            to_plot = [k for k in to_plot if 1 <= len(self._obj[k].dims) < 3]
            to_plot = to_plot[:5]

            if len(to_plot):
                fig, ax = plt.subplots(1, len(to_plot), figsize=(len(to_plot) * 3, 3,))
                if len(to_plot) == 1:
                    ax = [ax]

                for i, plot_var in enumerate(to_plot):
                    self._obj[plot_var].plot(ax=ax[i])
                    fancy_labels(ax[i])
                    ax[i].set_title(plot_var.replace('_', ' '))

                remove_colorbars()

        else:
            if 1 <= len(self._obj.dims) < 3:
                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                self._obj.plot(ax=ax)
                fancy_labels(ax)
                ax.set_title('')

                remove_colorbars()

        wrapper_style = 'style="display: flex; flex-direction: row;"'

        try:
            name = self.df_index
        except:
            if 'id' in self._obj.attrs:
                name = 'ID: ' + str(self._obj.attrs['id'])[:9] + '...'
            else:
                name = 'No name'

        warning = ''

        if len(self._obj.attrs) < 10:
            warning = ':  <span style="color: red;">Few Attributes, Data Is Summed?</span>'

        return """
        <header><strong>{name}{warning}</strong></header>
        <div {wrapper_style}>
        <details open>
            <summary>Experimental Conditions</summary>
            {conditions}
        </details>
        <details open>
            <summary>Full Coordinates</summary>
            {coordinates}
        </details>
        <details open>
            <summary>Spectrometer</summary>
            {spectrometer_info}
        </details>
        </div>
        """.format(
            name=name,
            warning=warning,
            wrapper_style=wrapper_style,
            conditions=self._repr_html_experimental_conditions(self.experimental_conditions),
            coordinates=self._repr_html_full_coords({k: v for k, v in self.full_coords.items() if v is not None}),
            spectrometer_info=self._repr_html_spectrometer_info(),
        )


@xr.register_dataarray_accessor('S')
class ARPESDataArrayAccessor(ARPESAccessorBase):
    def plot(self, *args, **kwargs):
        with plt.rc_context(rc={'text.usetex': False}):
            self._obj.plot(*args, **kwargs)

    def show(self, **kwargs):
        image_tool = ImageTool(**kwargs)
        return image_tool.make_tool(self._obj)

    def show_d2(self, **kwargs):
        curve_tool = CurvatureTool(**kwargs)
        return curve_tool.make_tool(self._obj)

    def show_band_tool(self, **kwargs):
        band_tool = BandTool(**kwargs)
        return band_tool.make_tool(self._obj)

    def fs_plot(self, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        if out is not None and isinstance(out, bool):
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


    def _referenced_scans_for_spatial_plot(self, use_id=True, pattern='{}.png', **kwargs):
        out = kwargs.get('out')
        label = self._obj.attrs['id'] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format('{}_reference_scan_fs'.format(label))
            kwargs['out'] = out

        return plotting.reference_scan_spatial(self._obj, **kwargs)


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

    def cut_nan_coords(self):
        slices = dict()
        for cname, cvalue in self._obj.coords.items():
            try:
                end_ind = np.where(np.isnan(cvalue.values))[0][0]
                end_ind = None if end_ind == -1 else end_ind
                slices[cname] = slice(None, end_ind)
            except IndexError:
                pass

        return self._obj.isel(**slices)

    def nan_to_num(self, x=0):
        """
        xarray version of numpy.nan_to_num
        :param x:
        :return:
        """

        data = self._obj.copy(deep=True)
        assert(isinstance(data, xr.DataArray))
        data.values[np.isnan(data.values)] = x
        return data

    def reference_plot(self, **kwargs):
        if self.spectrum_type == 'map':
            return self._referenced_scans_for_map_plot(**kwargs)
        elif self.spectrum_type == 'hv_map':
            return self._referenced_scans_for_hv_map_plot(**kwargs)
        elif self.spectrum_type == 'spectrum':
            return self._simple_spectrum_reference_plot(**kwargs)
        elif self.spectrum_type in {'ucut', 'spem'}:
            return self._referenced_scans_for_spatial_plot(**kwargs)
        elif self.spectrum_type in {'cut'}:
            return None
        else:
            import pdb
            pdb.set_trace()

    @property
    def eV(self):
        eV = None
        if 'eV' in self._obj.dims:
            eV = self._obj.eV
        else:
            for dof in set(self._obj.dims):
                if 'eV' in dof:
                    eV = self._obj[dof]
        return eV


NORMALIZED_DIM_NAMES = ['x', 'y', 'z', 'w']

@xr.register_dataset_accessor('T')
@xr.register_dataarray_accessor('T')
class GenericAccessorTools(object):
    _obj = None

    def extent(self, *args, dims=None):
        """
        Returns an "extent" array that can be used to draw with plt.imshow
        :return:
        """

        if dims is None:
            if len(args) == 0:
                dims = self._obj.dims
            else:
                dims = args

        assert(len(dims) == 2 and 'You must supply exactly two dims to `.T.extent` not {}'.format(dims))
        return [
            self._obj.coords[dims[0]][0].item(),
            self._obj.coords[dims[0]][-1].item(),
            self._obj.coords[dims[1]][0].item(),
            self._obj.coords[dims[1]][-1].item(),
        ]

    def drop_nan(self):
        assert(len(self._obj.dims) == 1)

        mask = np.logical_not(np.isnan(self._obj.values))
        return self._obj.isel(**dict([[self._obj.dims[0], mask]]))

    def shift_coords(self, dims, shift):
        if not isinstance(shift, np.ndarray):
            shift = np.ones((len(dims),)) * shift

        def transform(data):
            new_shift = shift
            for _ in range(len(dims)):
                new_shift = np.expand_dims(new_shift, 0)

            return data + new_shift

        return self.transform_coords(dims, transform)

    def scale_coords(self, dims, scale):
        if not isinstance(scale, np.ndarray):
            n_dims = len(dims)
            scale = np.identity(n_dims) * scale
        elif len(scale.shape) == 1:
            scale = np.diag(scale)

        return self.transform_coords(dims, scale)

    def transform_coords(self, dims, transform):
        """
        Transforms the given coordinate values according to transform, should either be a function
        from a len(dims) x size of raveled coordinate array to len(dims) x size of raveled_coordinate
        array or a linear transformation as a matrix which is multiplied into such an array.
        :param dims: List of dimensions that should be transformed
        :param transform: The transformation to apply, can either be a function, or a matrix
        :return:
        """
        as_array = np.stack([self._obj.data_vars[d].values for d in dims], axis=-1)

        if isinstance(transform, np.ndarray):
            transformed = np.dot(as_array, transform)
        else:
            transformed = transform(as_array)

        copy = self._obj.copy(deep=True)

        for d, arr in zip(dims, np.split(transformed, transformed.shape[-1], axis=-1)):
            copy.data_vars[d].values = np.squeeze(arr, axis=-1)

        return copy

    def filter_vars(self, f):
        return xr.Dataset(data_vars={
            k: v for k, v in self._obj.data_vars.items() if f(v, k)
        }, attrs=self._obj.attrs)

    def var_startswith(self, fragment):
        return self.filter_vars(lambda _, k: k.startswith(fragment))

    def var_contains(self, fragment):
        """
        Filters a dataset's variables based on whether the name contains the fragment
        :param fragment:
        :return:
        """
        return self.filter_vars(lambda _, k: fragment in k)

    def coordinatize(self, as_coordinate_name):
        """
        Remarkably, `coordinatize` is a word

        :return:
        """
        assert(len(self._obj.dims) == 1)

        d = self._obj.dims[0]
        o = self._obj.rename(dict([[d, as_coordinate_name]])).copy(deep=True)
        o.coords[as_coordinate_name] = o.values

        return o

    def ravel(self, as_dataset=False):
        """
        Converts to a flat representation where the coordinate values are also present.
        Extremely valuable for plotting a dataset with coordinates, X, Y and values Z(X,Y)
        on a scatter plot in 3D.

        By default the data is listed under the key 'data'.

        :return:
        """

        assert (isinstance(self._obj, xr.DataArray))

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        raveled_coordinates = dict(zip(dims, [cs.ravel() for cs in np.meshgrid(*coords_as_list)]))
        assert ('data' not in raveled_coordinates)
        raveled_coordinates['data'] = self._obj.values.ravel()

        return raveled_coordinates

    def meshgrid(self, as_dataset=False):
        assert (isinstance(self._obj, xr.DataArray))

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        meshed_coordinates = dict(zip(dims, [cs for cs in np.meshgrid(*coords_as_list)]))
        assert ('data' not in meshed_coordinates)
        meshed_coordinates['data'] = self._obj.values

        if as_dataset:
            # this could use a bit of cleaning up
            faked = ['x', 'y', 'z', 'w']
            meshed_coordinates = {k: (faked[:len(v.shape)], v) for k, v in meshed_coordinates.items() if k != 'data'}

            return xr.Dataset(meshed_coordinates)

        return meshed_coordinates

    def to_arrays(self):
        """
        Useful for rapidly converting into a format than can be `plt.scatter`ed
        or similar.

        Ex:

        plt.scatter(*data.T.as_arrays(), marker='s')
        :return:
        """
        assert(len(self._obj.dims) == 1)

        return [self._obj.coords[self._obj.dims[0]].values, self._obj.values]

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

    def map_axes(self, axes, fn, dtype=None, **kwargs):
        if isinstance(self._obj, xr.Dataset):
            raise TypeError('map_axes can only work on xr.DataArrays for now because of '
                            'how the type inference works')
        obj = self._obj.copy(deep=True)

        if dtype is not None:
            obj.values = np.ndarray(shape=obj.values.shape, dtype=dtype)

        type_assigned = False
        for coord, value in self.iterate_axis(axes):
            new_value = fn(value, coord)

            if dtype is None:
                if not type_assigned:
                    obj.values = np.ndarray(shape=obj.values.shape, dtype=new_value.data.dtype)
                    type_assigned = True

                obj.loc[coord] = new_value.values
            else:
                obj.loc[coord] = new_value

        return obj

    def map(self, fn, **kwargs):
        return apply_dataarray(self._obj, np.vectorize(fn, **kwargs))

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

    def stride(self, *args, generic_dim_names=True):
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_strides = [coord.values[1] - coord.values[0] for coord in indexed_coords]

        dim_names = self._obj.dims
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[:len(dim_names)]

        result = dict(zip(dim_names, indexed_strides))
        
        if len(args):
            if len(args) == 1:
                if not isinstance(args[0],str):
                    # if passed list of strs as argument
                    result = [result[selected_names] for selected_names in args[0]]
                else:
                    # if passed single name as argument
                    result = result[args[0]]
            else:
                # if passed several names as arguments
                result = [result[selected_names] for selected_names in args]
            
        return result

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

    def eval(self, *args, **kwargs):
        return self._obj.results.T.map(lambda x: x.eval(*args, **kwargs))

    def show(self):
        fit_diagnostic_tool = FitCheckTool()
        return fit_diagnostic_tool.make_tool(self._obj)

    def p(self, param_name):
        return self._obj.results.F.p(param_name)

    def s(self, param_name):
        return self._obj.results.F.s(param_name)

    def plot_param(self, param_name, **kwargs):
        return self._obj.results.F.plot_param(param_name, **kwargs)


@xr.register_dataarray_accessor('F')
class ARPESFitToolsAccessor(object):
    _obj = None

    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj

    def plot_param(self, param_name, **kwargs):
        plotting.plot_parameter(self._obj, param_name, **kwargs)

    def param_as_dataset(self, param_name):
        return xr.Dataset({
            'value': self.p(param_name),
            'error': self.s(param_name),
        })

    def show(self):
        fit_diagnostic_tool = FitCheckTool()
        return fit_diagnostic_tool.make_tool(self._obj)

    def show_fit_diagnostic(self):
        return self.show()

    def p(self, param_name):
        return self._obj.T.map(param_getter(param_name), otypes=[np.float])

    def s(self, param_name):
        return self._obj.T.map(param_stderr_getter(param_name), otypes=[np.float])

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
    def is_spatial(self):
        try:
            return self.spectrum.S.is_spatial
        except Exception as e:
            return False

    @property
    def spectrum(self) -> Optional[xr.DataArray]:
        spectrum = None
        if 'spectrum' in self._obj.data_vars:
            spectrum = self._obj.spectrum
        elif 'raw' in self._obj.data_vars:
            spectrum = self._obj.raw
        elif '__xarray_dataarray_variable__' in self._obj.data_vars:
            spectrum = self._obj.__xarray_dataarray_variable__
        elif any('spectrum' in dv for dv in self._obj.data_vars):
            spectrum = self._obj[list(self._obj.data_vars)[list('spectrum' in dv for dv in self._obj.data_vars).index(True)]]

        if spectrum is not None and 'df' not in spectrum.attrs:
            spectrum.attrs['df'] = self._obj.attrs.get('df', None)

        return spectrum

    @property
    def spectra(self):
        spectra = []
        for dv in list(self._obj.data_vars):
            if 'spectrum' in dv:
                spectra.append(self._obj[dv])

        return spectra

    @property
    def is_multi_region(self):
        return len(self.spectra) > 1

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
        4. For spatial scans:
           i. energy/angle integrated spatial maps with subsequent measurements indicated
           2. energy/angle integrated FS spatial maps with subsequent measurements indicated

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

        if self.is_spatial:
            referenced = self.referenced_scans

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



