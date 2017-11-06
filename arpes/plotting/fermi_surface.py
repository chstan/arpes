import holoviews as hv
import xarray as xr

from arpes.provenance import save_plot_provenance
from .utils import path_for_plot, path_for_holoviews

__all__ = ['fermi_surface_slices']

@save_plot_provenance
def fermi_surface_slices(arr: xr.DataArray, n_slices=9, ev_per_slice=0.02, bin=0.01, out=None, **kwargs):
    slices = []
    for i in range(n_slices):
        high = - ev_per_slice * i
        low = high - bin
        image = hv.Image(arr.sum([d for d in arr.dims if d not in ['polar', 'phi', 'eV', 'kp', 'kx', 'ky']]).sel(
            eV=slice(low, high)).sum('eV'), label='%g eV' % high)

        slices.append(image)

    layout = hv.Layout(slices).cols(3)
    if out is not None:
        renderer = hv.renderer('matplotlib').instance(fig='svg', holomap='gif')
        filename = path_for_plot(out)
        renderer.save(layout, path_for_holoviews(filename))
        return filename
    else:
        return layout

