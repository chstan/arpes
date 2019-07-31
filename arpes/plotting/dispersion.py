import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import xarray as xr
from arpes.utilities import bz
from arpes.io import load_dataset
from arpes.utilities.conversion import remap_coords_to
from arpes.preparation import normalize_dim

import warnings

from arpes.provenance import save_plot_provenance
from .utils import *

__all__ = ['plot_dispersion', 'labeled_fermi_surface',
           'cut_dispersion_plot', 'fancy_dispersion', 'reference_scan_fermi_surface',
           'hv_reference_scan', 'scan_var_reference_plot']


def band_path(band):
    import holoviews as hv
    return hv.Path([band.center.values, band.coords[band.dims[0]].values])


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
def cut_dispersion_plot(data: xr.DataArray, e_floor=None, title=None, ax=None, include_symmetry_points=True,
                        out=None, quality='high', **kwargs):
    """
    Makes a 3D cut dispersion plot. At the moment this only supports rectangular BZs.
    :param data:
    :param e_floor:
    :param title:
    :param ax:
    :param out:
    :param kwargs:
    :return:
    """

    # to get nice labeled edges you could use shapely
    sampling = {
        'paper': 400,
        'high': 100,
        'low': 40,
    }.get(quality, 100)

    assert('eV' in data.dims)
    assert(e_floor is not None)

    new_dim_order = list(data.dims)
    new_dim_order.remove('eV')
    new_dim_order = new_dim_order + ['eV']
    data = data.transpose(*new_dim_order)

    # prep data to be used for rest of cuts
    lower_part = data.sel(eV=slice(None, 0))
    floor = lower_part.S.fat_sel(eV=e_floor)

    bz_mask = bz.reduced_bz_mask(lower_part, scale_zone=True)
    left_mask = bz.reduced_bz_E_mask(lower_part, 'X', e_floor, scale_zone=True)
    right_mask = bz.reduced_bz_E_mask(lower_part, 'Y', e_floor, scale_zone=True)
    mask_for = lambda x: left_mask if x.shape == left_mask.shape else right_mask

    x_dim, y_dim, z_dim = tuple(new_dim_order)
    x_coords, y_coords, z_coords = data.coords[x_dim], data.coords[y_dim], data.coords[z_dim]

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    if title is None:
        title = '{} Cut Through Symmetry Points'.format(data.S.label.replace('_', ' '))

    ax.set_title(title)
    colormap = plt.get_cmap('Blues')

    # color fermi surface
    fermi_surface = data.S.fermi_surface
    Xs, Ys = np.meshgrid(x_coords, y_coords)

    Zs = np.zeros(fermi_surface.data.shape)
    Zs[bz_mask] = np.nan
    scale_colors = max(np.max(fermi_surface.data), np.max(floor.data))
    ax.plot_surface(Xs, Ys, Zs.T, facecolors=colormap(fermi_surface.data.T / scale_colors), shade=False,
                    vmin=-1, vmax=1, antialiased=True, rcount=sampling, ccount=sampling)

    # color right edge
    right_sel = dict()
    edge_val = np.max(lower_part.coords[x_dim].data)
    right_sel[x_dim] = edge_val
    right_edge = lower_part.S.fat_sel(**right_sel)

    Ys, Zs = np.meshgrid(lower_part.coords[y_dim], lower_part.coords[z_dim])
    Xs = np.ones(right_edge.shape) * edge_val
    Xs[mask_for(Xs)] = np.nan
    ax.plot_surface(Xs.T, Ys, Zs, facecolors=colormap(right_edge.data.T / scale_colors),
                    vmin=-1, vmax=1, shade=False, antialiased=True, rcount=sampling, ccount=sampling)

    # color left edge
    left_sel = dict()
    edge_val = np.min(lower_part.coords[y_dim].data)
    left_sel[y_dim] = edge_val
    left_edge = lower_part.S.fat_sel(**left_sel)
    max_left_edge = np.max(left_edge.data)

    Xs, Zs = np.meshgrid(lower_part.coords[x_dim], lower_part.coords[z_dim])
    Ys = np.ones(left_edge.shape) * edge_val
    Ys[mask_for(Ys)] = np.nan
    ax.plot_surface(Xs, Ys.T, Zs, facecolors=colormap(left_edge.data.T / scale_colors),
                    vmin=-1, vmax=1, antialiased=True, shade=False, rcount=sampling, ccount=sampling)

    # selection region
    # floor
    Xs, Ys = np.meshgrid(floor.coords[x_dim], floor.coords[y_dim])
    Zs = np.ones(floor.data.shape) * e_floor
    Zs[np.logical_not(bz_mask)] = np.nan
    ax.plot_surface(Xs, Ys, Zs.T, facecolors=colormap(floor.data.T / scale_colors),
                    vmin=-1, vmax=1, shade=False, antialiased=True, rcount=sampling, ccount=sampling)

    # determine the axis along X, Y
    axis_X = bz.axis_along(data, 'X')
    axis_Y = bz.axis_along(data, 'Y')

    # left and right inset faces
    inset_face = lower_part.S.along(['G', 'X'], axis_name=axis_X, extend_to_edge=True).sel(eV=slice(e_floor, None))
    Xs, Zs = np.meshgrid(inset_face.coords[axis_X], inset_face.coords[z_dim])
    Ys = np.ones(inset_face.data.shape)
    if y_dim == axis_X:
        Ys *= data.S.phi_offset
        Xs += data.S.map_angle_offset
        Xs, Ys = Ys, Xs
    else:
        Ys *= data.S.map_angle_offset
        Xs += data.S.phi_offset
    ax.plot_surface(Xs, Ys, Zs, facecolors=colormap(inset_face.data / scale_colors),
                    shade=False, antialiased=True, zorder=1, rcount=sampling, ccount=sampling)

    inset_face = lower_part.S.along(['G', 'Y'], axis_name=axis_Y, extend_to_edge=True).sel(eV=slice(e_floor, None))
    Ys, Zs = np.meshgrid(inset_face.coords[axis_Y], inset_face.coords[z_dim])
    Xs = np.ones(inset_face.data.shape)
    if x_dim == axis_Y:
        Xs *= data.S.map_angle_offset
        Ys += data.S.phi_offset
        Xs, Ys = Ys, Xs
    else:
        Xs *= data.S.phi_offset
        Ys += data.S.map_angle_offset

    ax.plot_surface(Xs, Ys, Zs, facecolors=colormap(inset_face.data / scale_colors),
                    shade=False, antialiased=True, zorder=1, rcount=sampling, ccount=sampling)

    ax.set_xlabel(label_for_dim(data, x_dim))
    ax.set_ylabel(label_for_dim(data, y_dim))
    ax.set_zlabel(label_for_dim(data, z_dim))

    zlim = ax.get_zlim3d()
    if include_symmetry_points:
        for point_name, point_location in data.S.iter_symmetry_points:
            coords = [point_location.get(d, 0.02) for d in new_dim_order]
            ax.scatter(*zip(coords), marker='.', color='red', zorder=1000)
            coords[new_dim_order.index('eV')] += 0.1
            ax.text(*coords, label_for_symmetry_point(point_name), color='red', ha='center', va='top')

    ax.set_zlim3d(*zlim)
    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()


@save_plot_provenance
def hv_reference_scan(data, out=None, e_cut=-0.05, bkg_subtraction=0.8, **kwargs):
    fs = data.S.fat_sel(eV=e_cut)
    fs = normalize_dim(fs, 'hv', keep_id=True)
    fs.data -= bkg_subtraction * np.mean(fs.data)
    fs.data[fs.data < 0] = 0

    fig, ax = labeled_fermi_surface(fs, hold=True, **kwargs)

    all_scans = data.attrs['df']
    all_scans = all_scans[all_scans.id != data.attrs['id']]
    all_scans = all_scans[(all_scans.spectrum_type != 'xps_spectrum') |
                          (all_scans.spectrum_type == 'hv_map')]

    scans_by_hv = defaultdict(list)
    for index, row in all_scans.iterrows():
        scan = load_dataset(row.id)

        scans_by_hv[round(scan.S.hv)].append(scan.S.label.replace('_', ' '))

    dim_order = ax.dim_order
    handles = []
    handle_labels = []

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for line_color, (hv, labels) in zip(colors, scans_by_hv.items()):
        full_label = '\n'.join(labels)

        # determine direction
        if dim_order[0] == 'hv':
            # photon energy is along the x axis, we want an axvline
            handles.append(ax.axvline(hv, label=full_label, color=line_color))
        else:
            # photon energy is along the y axis, we want an axhline
            handles.append(ax.axhline(hv, label=full_label, color=line_color))

        handle_labels.append(full_label)

    plt.legend(handles, handle_labels)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()

@save_plot_provenance
def reference_scan_fermi_surface(data, out=None, **kwargs):
    fs = data.S.fermi_surface
    fig, ax = labeled_fermi_surface(fs, hold=True, **kwargs)

    referenced_scans = data.S.referenced_scans
    handles = []
    for index, row in referenced_scans.iterrows():
        scan = load_dataset(row.id)
        remapped_coords = remap_coords_to(scan, data)
        dim_order = ax.dim_order
        ls = ax.plot(remapped_coords[dim_order[0]], remapped_coords[dim_order[1]], label=index.replace('_', ' '))
        handles.append(ls[0])

    plt.legend(handles=handles)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()


@save_plot_provenance
def labeled_fermi_surface(data, title=None, ax=None, hold=False,
                          include_symmetry_points=True, include_bz=True,
                          out=None, fermi_energy=0, **kwargs):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if title is None:
        title = '{} Fermi Surface'.format(data.S.label.replace('_', ' '))

    if 'eV' in data.dims:
        data = data.S.generic_fermi_surface(fermi_energy)

    mesh = data.plot(ax=ax)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap('Blues')

    dim_order = [ax.get_xlabel(), ax.get_ylabel()]

    setattr(ax, 'dim_order', dim_order)
    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))
    ax.set_title(title)

    marker_color = 'red' if data.S.is_differentiated else 'red'

    if include_bz:
        symmetry = bz.bz_symmetry(data.S.iter_own_symmetry_points)

        # TODO Implement this
        warnings.warn('BZ region display not implemented.')

    if include_symmetry_points:
        for point_name, point_location in data.S.iter_symmetry_points:
            warnings.warn('Symmetry point locations are not k-converted')
            coords = [point_location[d] for d in dim_order]
            ax.plot(*coords, marker='.', color=marker_color)
            ax.annotate(label_for_symmetry_point(point_name), coords, color=marker_color,
                        xycoords='data', textcoords='offset points', xytext=(0, -10),
                        va='top', ha='center', fontsize=14)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    if not hold:
        plt.show()
    else:
        return fig, ax


@save_plot_provenance
def fancy_dispersion(data, title=None, ax=None, out=None, include_symmetry_points=True,
                     norm=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if title is None:
        title = data.S.label.replace('_', ' ')

    mesh = data.plot(norm=norm, ax=ax, **kwargs)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap('Blues')

    original_x_label = ax.get_xlabel()
    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))
    ax.set_title(title, fontsize=14)

    # This can probably be pulled out into a a helper
    marker_color = 'red' if data.S.is_differentiated else 'red'
    if include_symmetry_points:
        for point_name, point_locations in data.attrs.get('symmetry_points', {}).items():
            if not isinstance(point_locations, list):
                point_locations = [point_locations]
            for single_location in point_locations:
                coords = (single_location[original_x_label], ax.get_ylim()[1],)
                ax.plot(*coords, marker=11, color=marker_color)
                ax.annotate(label_for_symmetry_point(point_name), coords, color=marker_color,
                            xycoords='data', textcoords='offset points', xytext=(0, -10),
                            va='top', ha='center')


    ax.axhline(0, color='red', alpha=0.8, linestyle='--', linewidth=1)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()
    return ax


@save_plot_provenance
def scan_var_reference_plot(data, title=None, ax=None, norm=None, out=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if title is None:
        title = data.S.label.replace('_', ' ')

    plot = data.plot(norm=norm, ax=ax)
    plot.colorbar.set_label(label_for_colorbar(data))

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))

    ax.set_title(title, font_size=14)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()