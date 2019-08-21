from typing import Any, Dict, List, Optional
import itertools

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec

from arpes.io import simple_load
from arpes.plotting import annotate_point
from arpes.plotting.utils import (path_for_plot, frame_with,
                                  remove_colorbars, fancy_labels,
                                  ddata_daxis_units)
from arpes.provenance import save_plot_provenance
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.xarray import unwrap_xarray_item


__all__ = ('reference_scan_spatial', 'plot_spatial_reference')


@save_plot_provenance
def plot_spatial_reference(
        reference_map: DataType, data_list: List[DataType],
        offset_list: Optional[List[Dict[str, Any]]] = None,
        annotation_list: Optional[List[str]] = None,
        out: Optional[str] = None, plot_refs: bool = True):
    """
    Helpfully plots data against a reference scanning dataset. This is essential to understand
    where data was taken and can be used early in the analysis phase in order to highlight the
    location of your datasets against core levels, etc.

    :param reference_map: A scanning photoemission like dataset
    :param data_list: A list of datasets you want to plot the relative locations of
    :param offset_list: Optionally, offsets given as coordinate dicts
    :param annotation_list: Optionally, text annotations for the data
    :param out: Where to save the figure if we are outputting to disk
    :param plot_refs: Whether to plot reference figures for each of the pieces of data in `data_list`
    :return:
    """
    if offset_list is None:
        offset_list = [None] * len(data_list)

    if annotation_list is None:
        annotation_list = [str(i + 1) for i in range(len(data_list))]

    normalize_to_spectrum(reference_map)

    n_references = len(data_list)
    if n_references == 1 and plot_refs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5,))
        ax = axes[0]
        ax_refs = [axes[1]]
    elif plot_refs:
        n_extra_axes = 1 + (n_references // 4)
        fig = plt.figure(figsize=(6 * (1 + n_extra_axes), 5,))
        spec = gridspec.GridSpec(ncols=2 * (1 + n_extra_axes), nrows=2, figure=fig)
        ax = fig.add_subplot(spec[:2, :2])

        ax_refs = [fig.add_subplot(spec[i // (2 * n_extra_axes), 2 + i % (2 * n_extra_axes)])
                   for i in range(n_references)]
    else:
        ax_refs = []
        fig, ax = plt.subplots(1, 1, figsize=(6, 5,))

    try:
        reference_map = reference_map.S.spectra[0]
    except Exception:
        pass

    reference_map = reference_map.S.mean_other(['x', 'y', 'z'])

    ref_dims = reference_map.dims[::-1]

    assert len(reference_map.dims) == 2
    reference_map.S.plot(ax=ax, cmap='Blues')

    cmap = cm.get_cmap('Reds')
    rendered_annotations = []
    for i, (data, offset, annotation) in enumerate(zip(data_list, offset_list, annotation_list)):
        if offset is None:
            try:
                offset = data.S.logical_offsets - reference_map.S.logical_offsets
            except ValueError:
                offset = {}

        coords = {c: unwrap_xarray_item(data.coords[c]) for c in ref_dims}
        n_array_coords = len([cv for cv in coords.values()
                              if isinstance(cv, (np.ndarray, xr.DataArray))])
        color = cmap(0.4 + (0.5 * i / len(data_list)))
        x = coords[ref_dims[0]] + offset.get(ref_dims[0], 0)
        y = coords[ref_dims[1]] + offset.get(ref_dims[1], 0)
        ref_x, ref_y = x, y
        off_x, off_y = 0, 0
        scale = 0.03

        if n_array_coords == 0:
            off_y = 1
            ax.scatter([x], [y], s=60, color=color)
        if n_array_coords == 1:
            if isinstance(x, (np.ndarray, xr.DataArray)):
                y = [y] * len(x)
                ref_x = np.min(x)
                off_x = -1
            else:
                x = [x] * len(y)
                ref_y = np.max(y)
                off_y = 1

            ax.plot(x, y, color=color, linewidth=3)
        if n_array_coords == 2:
            off_y = 1
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            ref_x, ref_y = min_x, max_y

            color = cmap(0.4 + (0.5 * i / len(data_list)), alpha=0.5)
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, facecolor=color)
            color = cmap(0.4 + (0.5 * i / len(data_list)))

            ax.add_patch(rect)

        dp = ddata_daxis_units(ax)
        text_location = np.asarray([ref_x, ref_y, ]) + dp * scale * np.asarray([off_x, off_y])
        text = ax.annotate(annotation, text_location, color='black', size=15)
        rendered_annotations.append(text)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                               path_effects.Normal()])
        if plot_refs:
            ax_ref = ax_refs[i]
            keep_preference = list(ref_dims) + [
                'eV', 'temperature', 'kz', 'hv', 'kp', 'kx', 'ky', 'phi', 'theta', 'beta',
                'pixel',
            ]
            keep = [d for d in keep_preference if d in data.dims][:2]
            data.S.mean_other(keep).S.plot(ax=ax_ref)
            ax_ref.set_title(annotation)
            fancy_labels(ax_ref)
            frame_with(ax_ref, color=color, linewidth=3)

    ax.set_title('')
    remove_colorbars()
    fancy_labels(ax)
    plt.tight_layout()

    try:
        from adjustText import adjust_text
        adjust_text(rendered_annotations, ax=ax,
                    avoid_points=False, avoid_objects=False,
                    avoid_self=False, autoalign='xy')
    except ImportError:
        pass

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, [ax] + ax_refs


@save_plot_provenance
def reference_scan_spatial(data, out=None, **kwargs):
    data = normalize_to_spectrum(data)

    dims = [d for d in data.dims if d in {'cycle', 'phi', 'eV'}]

    summed_data = data.sum(dims, keep_attrs=True)

    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    flat_axes = list(itertools.chain(*ax))

    summed_data.plot(ax=flat_axes[0])
    flat_axes[0].set_title(r'Full \textbf{eV} range')

    dims_except_eV = [d for d in dims if d != 'eV']
    summed_data = data.sum(dims_except_eV)

    mul = 0.2
    rng = data.coords['eV'].max().item() - data.coords['eV'].min().item()
    offset = data.coords['eV'].max().item()
    if offset > 0:
        offset = 0

    if rng > 3:
        mul = rng / 5.

    for i in range(5):
        low_e, high_e = -mul * (i + 1) + offset, -mul * i + offset
        title = r'\textbf{eV}' + ': {:.2g} to {:.2g}'.format(low_e, high_e)
        summed_data.sel(eV=slice(low_e, high_e)).sum('eV').plot(ax=flat_axes[i+1])
        flat_axes[i + 1].set_title(title)

    y_range = flat_axes[0].get_ylim()
    x_range = flat_axes[0].get_xlim()
    delta_one_percent = ((x_range[1] - x_range[0]) / 100, (y_range[1] - y_range[0]) / 100)

    smart_delta = (2 * delta_one_percent[0], -1.5 * delta_one_percent[0])

    referenced = data.S.referenced_scans

    # idea here is to collect points by those that are close together, then
    # only plot one annotation
    condensed = []
    cutoff = 3 # 3 percent
    for index, row in referenced.iterrows():
        ff = simple_load(index)

        x, y, _ = ff.S.sample_pos
        found = False
        for cx, cy, cl in condensed:
            if abs(cx  - x) < cutoff * abs(delta_one_percent[0]) and abs(cy - y) < cutoff * abs(delta_one_percent[1]):
                cl.append(index)
                found = True
                break

        if not found:
            condensed.append((x, y, [index]))

    for fax in flat_axes:
        for cx, cy, cl in condensed:
            annotate_point(fax, (cx, cy,), ','.join([str(l) for l in cl]),
                           delta=smart_delta, fontsize='large')

    plt.tight_layout()

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
