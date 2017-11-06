import holoviews as hv
import matplotlib.pyplot as plt
import xarray as xr

from arpes.plotting.utils import path_for_plot
from arpes.provenance import save_plot_provenance

__all__ = ['plot_dispersion', 'plot_dispersion_holoview']

def band_path(band):
    return hv.Path([band.center.values, band.coords[band.dims[0]].values])

@save_plot_provenance
def plot_dispersion_holoview(spectrum: xr.DataArray, bands, out=None):
    image = hv.Image(spectrum)
    band = band_path(bands[1])

    # TODO consider refactoring the save code into the decorator, this might not always
    # make sense, but the plotting function can opt out by returning a string instead and
    # doing the plotting anyway
    if out is not None:
        renderer = hv.renderer('matplotlib').instance(fig='svg', holomap='gif')
        filename = path_for_plot(out)
        renderer.save(band, filename)
    else:
        return band


@save_plot_provenance
def plot_dispersion(spectrum: xr.DataArray, bands, out=None):
    ax = spectrum.plot()

    for band in bands:
        plt.scatter(band.center.values, band.coords[band.dims[0]].values)

    if out is not None:
        filename = path_for_plot(out)
        plt.savefig(filename)
        return filename
    else:
        return ax