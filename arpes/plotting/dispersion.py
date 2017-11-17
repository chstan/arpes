import holoviews as hv
import matplotlib.pyplot as plt
import xarray as xr

from arpes.provenance import save_plot_provenance
from .utils import *

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


@save_plot_provenance
def labeled_fermi_surface(data, title=None, ax=None, out=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if title is None:
        title = '{} Fermi Surface'.format(data.S.label)

    mesh = data.plot(norm=kwargs.get('norm'), ax=ax)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap('Blues')

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))
    ax.set_title(title,)



@save_plot_provenance
def fancy_dispersion(data, title=None, ax=None, out=None, norm=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if title is None:
        title = data.S.label

    mesh = data.plot(norm=norm, ax=ax)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap('Blues')

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))
    ax.set_title(title, fontsize=14)

    # This can probably be pulled out into a a helper
    marker_color = 'red' if data.S.is_differentiated else 'red'
    for point_name, point_locations in data.attrs.get('symmetry_points', {}).items():
        if not isinstance(point_locations, list):
            point_locations = [point_locations]
        for single_location in point_locations:
            coords = (single_location['phi'], ax.get_ylim()[1],)
            ax.plot(*coords, marker=11, color=marker_color)
            ax.annotate(label_for_symmetry_point(point_name), coords, color=marker_color,
                        xycoords='data', textcoords='offset points', xytext=(0, -10),
                        va='top', ha='center')


    ax.axhline(0, color='red', alpha=0.8, linestyle='--', linewidth=1)

    plt.show()
    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)
