import warnings

import xarray as xr

import arpes.constants
import arpes.materials
from arpes.plotting import make_bokeh_tool, make_curvature_tool

__all__ = ['ARPESDataArrayAccessor', 'ARPESDatasetAccessor']


class _ARPESAccessorBase(object):
    """
    Base class for the xarray extensions that we put onto our datasets to make working with ARPES data a
    little cleaner. This allows you to access common attributes
    """
    @property
    def hv(self):
        if 'hv' in self._obj.attrs:
            return float(self._obj.attrs['hv'])

        if 'location' in self._obj.attrs:
            if self._obj.attrs['location'] == 'ALG-MC':
                return 5.93

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
        return self._obj.sel(eV=0, method='nearest')

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataarray_accessor('S')
class ARPESDataArrayAccessor(_ARPESAccessorBase):
    def show(self):
        return make_bokeh_tool(self._obj)

    def show_d2(self):
        return make_curvature_tool(self._obj)


@xr.register_dataset_accessor('S')
class ARPESDatasetAccessor(_ARPESAccessorBase):
    def __init__(self, xarray_obj):
        super(ARPESDatasetAccessor, self).__init__(xarray_obj)
        self._spectrum = None

        data_arr_names = self._obj.data_vars.keys()

        spectrum_candidates = ['raw', 'spectrum', 'spec']
        if len(data_arr_names) == 1:
            self._spectrum = self._obj.data_vars[data_arr_names[0]]
        else:
            for candidate in spectrum_candidates:
                if candidate in data_arr_names:
                    self._spectrum = self._obj.data_vars[candidate]
                    break

        if self._spectrum is None:
            assert(False and "Dataset appears not to contain a spectrum among options {}".format(data_arr_names))