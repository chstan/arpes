import collections
import contextlib
import datetime
import errno
import itertools
import json
import os.path
import warnings
import pathlib
from collections import Counter

import matplotlib
import matplotlib.cm as cm
import matplotlib.offsetbox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colorbar, colors, gridspec
from matplotlib.lines import Line2D

import xarray as xr
from arpes import VERSION
from arpes.config import CONFIG, SETTINGS
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.jupyter import get_recent_history, get_notebook_name

__all__ = (
    # General + IO
    'path_for_plot', 'path_for_holoviews', 'name_for_dim', 'unit_for_dim',
    'savefig', 'AnchoredHScaleBar', 'calculate_aspect_ratio',

    # context managers
    'dark_background',

    # color related
    'temperature_colormap',
    'polarization_colorbar',
    'temperature_colormap_around',
    'temperature_colorbar',
    'temperature_colorbar_around',

    'generic_colorbarmap',
    'generic_colorbarmap_for_data',
    'colorbarmaps_for_axis',

    # Axis generation
    'dos_axes',
    'simple_ax_grid',

    # matplotlib 'macros'
    'invisible_axes',
    'no_ticks',
    'get_colorbars',
    'remove_colorbars',
    'frame_with',

    'imshow_arr',
    'imshow_mask',
    'lineplot_arr',  # 1D version of imshow_arr
    'plot_arr',  # generic dimension version of imshow_arr, plot_arr

    # insets related
    'inset_cut_locator',
    'swap_xaxis_side',
    'swap_yaxis_side',
    'swap_axis_sides',

    # units related
    'data_to_axis_units',
    'axis_to_data_units',
    'daxis_ddata_units',
    'ddata_daxis_units',

    # TeX related
    'quick_tex',

    # Decorating + labeling
    'label_for_colorbar', 'label_for_dim', 'label_for_symmetry_point',
    'sum_annotation',
    'mean_annotation',
    'fancy_labels',

    # Data summaries
    'summarize',

    'transform_labels',

    'v_gradient_fill',
    'h_gradient_fill',
)

def h_gradient_fill(x1, x2, x_solid, fill_color=None, ax=None, zorder=None, alpha=None, **kwargs):
    """
    Fills a gradient between x1 and x2. If x_solid is not None, the gradient will be extended
    at the maximum opacity from the closer limit towards x_solid.

    :param x1:
    :param x2:
    :param x_solid:
    :param fill_color:
    :param ax:
    :param zorder:
    :param alpha:
    :param kwargs:
    :return:
    """
    if ax is None:
        ax = plt.gca()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert fill_color

    alpha = 1.0 if alpha is None else alpha

    z = np.empty((1, 100, 4), dtype=float)

    rgb = matplotlib.colors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[None, :]

    xmin, xmax, (ymin, ymax) = x1, x2, ylim
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    if x_solid is not None:
        xlow, xhigh = (x2, x_solid) if x_solid > x2 else (x_solid, x1)
        ax.fill_betweenx(ylim, xlow, xhigh, color=fill_color, alpha=alpha)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im


def v_gradient_fill(y1, y2, y_solid, fill_color=None, ax=None, zorder=None, alpha=None, **kwargs):
    """
    Fills a gradient vertically between y1 and y2. If y_solid is not None, the gradient will be extended
    at the maximum opacity from the closer limit towards y_solid.

    :param y1:
    :param y2:
    :param y_solid:
    :param fill_color:
    :param ax:
    :param zorder:
    :param alpha:
    :param kwargs:
    :return:
    """
    if ax is None:
        ax = plt.gca()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert fill_color

    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)

    rgb = matplotlib.colors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    (xmin, xmax), ymin, ymax = xlim, y1, y2
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    if y_solid is not None:
        ylow, yhigh = (y2, y_solid) if y_solid > y2 else (y_solid, y1)
        ax.fill_between(xlim, ylow, yhigh, color=fill_color, alpha=alpha)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im

def simple_ax_grid(n_axes, figsize=None, **kwargs):
    """
    Generates a square-ish set of axes and hides the extra ones

    It would be nice to accept an "aspect ratio" item that will attempt to fix the
    grid dimensions to get an aspect ratio close to the desired one.

    :param n_axes:
    :param figsize:
    :param kwargs:
    :return:
    """
    width = int(np.ceil(np.sqrt(n_axes)))
    height = width - 1
    if width * height < n_axes:
        height += 1

    if figsize is None:
        figsize = (3 * max(width, 5), 3 * max(height, 5),)

    fig, ax = plt.subplots(height, width, figsize=figsize, **kwargs)
    if n_axes == 1:
        ax = np.array([ax])

    ax, ax_rest = ax.ravel()[:n_axes], ax.ravel()[n_axes:]
    for axi in ax_rest:
        invisible_axes(axi)

    return fig, ax, ax_rest


@contextlib.contextmanager
def dark_background(overrides):
    defaults = {
        'axes.edgecolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white', 'axes.labelcolor': 'white',
        'text.color': 'white'
    }
    defaults.update(overrides)

    with plt.rc_context(defaults):
        yield


def data_to_axis_units(points, ax=None):
    if ax is None:
        ax = plt.gca()

    return ax.transAxes.inverted().transform(ax.transData.transform(points))


def axis_to_data_units(points, ax=None):
    if ax is None:
        ax = plt.gca()

    return ax.transData.inverted().transform(ax.transAxes.transform(points))


def ddata_daxis_units(ax=None):
    if ax is None:
        ax = plt.gca()

    dp1 = axis_to_data_units((1., 1.), ax)
    dp0 = axis_to_data_units((0., 0.), ax)
    return dp1 - dp0


def daxis_ddata_units(ax=None):
    if ax is None:
        ax = plt.gca()

    dp1 = data_to_axis_units((1., 1.), ax)
    dp0 = data_to_axis_units((0., 0.), ax)
    return dp1 - dp0


def swap_xaxis_side(ax):
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


def swap_yaxis_side(ax):
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def swap_axis_sides(ax):
    swap_xaxis_side(ax)
    swap_yaxis_side(ax)


def transform_labels(transform_fn, fig=None, include_titles=True):
    if fig is None:
        fig = plt.gcf()

    axes = list(fig.get_axes())
    for ax in axes:
        try:
            ax.set_xlabel(transform_fn(ax.get_xlabel(), is_title=False))
            ax.set_ylabel(transform_fn(ax.get_xlabel(), is_title=False))
            if include_titles:
                ax.set_title(transform_fn(ax.get_title(), is_title=True))
        except TypeError:
            ax.set_xlabel(transform_fn(ax.get_xlabel()))
            ax.set_ylabel(transform_fn(ax.get_xlabel()))
            if include_titles:
                ax.set_title(transform_fn(ax.get_title()))


def summarize(data: DataType, axes=None):
    data = normalize_to_spectrum(data)

    axes_shapes_for_dims = {
        1: (1, 1),
        2: (1, 1),
        3: (2, 2),  # one extra here
        4: (3, 2),  # corresponds to 4 choose 2 axes
    }

    if axes is None:
        fig, axes = plt.subplots(axes_shapes_for_dims.get(len(data.dims)), figsize=(8, 8))

    flat_axes = axes.ravel()
    combinations = list(itertools.combinations(data.dims, 2))
    for axi, combination in zip(flat_axes, combinations):
        data.sum(combination).plot(ax=axi)
        fancy_labels(axi)

    for i in range(len(combinations), len(flat_axes)):
        flat_axes[i].set_axis_off()

    return axes


def sum_annotation(eV=None, phi=None):
    eV_annotation, phi_annotation = '', ''

    def to_str(bound):
        if bound is None:
            return ''

        return '{:.2f}'.format(bound)

    if eV is not None:
        if SETTINGS['use_tex']:
            eV_annotation = '$\\text{E}_{' + to_str(eV.start) + '}^{' + to_str(eV.stop) + '}$'
        else:
            eV_annotation = to_str(eV.start) + ' < E < ' + to_str(eV.stop)
    if phi is not None:
        if SETTINGS['use_tex']:
            phi_annotation = '$\\phi_{' + to_str(phi.start) + '}^{' + to_str(phi.stop) + '}$'
        else:
            phi_annotation = to_str(phi.start) + ' < φ < ' + to_str(phi.stop)

    return eV_annotation + phi_annotation


def mean_annotation(eV=None, phi=None):
    eV_annotation, phi_annotation = '', ''

    def to_str(bound):
        if bound is None:
            return ''

        return '{:.2f}'.format(bound)

    if eV is not None:
        if SETTINGS['use_tex']:
            eV_annotation = '$\\bar{\\text{E}}_{' + to_str(eV.start) + '}^{' + to_str(eV.stop) + '}$'
        else:
            eV_annotation = 'Mean<' + to_str(eV.start) + ' < E < ' + to_str(eV.stop) + '>'
    if phi is not None:
        if SETTINGS['use_tex']:
            phi_annotation = '$\\bar{\\phi}_{' + to_str(phi.start) + '}^{' + to_str(phi.stop) + '}$'
        else:
            phi_annotation = 'Mean<' + to_str(phi.start) + ' < φ < ' + to_str(phi.stop) + '>'

    return eV_annotation + phi_annotation


def frame_with(ax, color='red', linewidth=2):
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(linewidth)


def quick_tex(latex_fragment, ax=None, fontsize=30):
    """
    Sometimes you just need to render some latex and getting a latex session
    running is far too much effort.
    :param latex_fragment:
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()

    invisible_axes(ax)
    ax.text(0.2, 0.2, latex_fragment, fontsize=fontsize)
    return ax


def lineplot_arr(arr, ax=None, method='plot', mask=None, mask_kwargs=None, **kwargs):
    if mask_kwargs is None:
        mask_kwargs = dict()

    if ax is None:
        _, ax = plt.subplots()

    xs = None
    if arr is not None:
        fn = plt.plot
        if method == 'scatter':
            fn = plt.scatter

        xs = arr.coords[arr.dims[0]].values
        fn(xs, arr.values, **kwargs)

    if mask is not None:
        y_lim = ax.get_ylim()
        if isinstance(mask, list) and isinstance(mask[0], slice):
            for slice_mask in mask:
                ax.fill_betweenx(y_lim, slice_mask.start, slice_mask.stop, **mask_kwargs)
        else:
            raise NotImplementedError()
        ax.set_ylim(y_lim)

    return ax


def plot_arr(arr=None, ax=None, over=None, mask=None, **kwargs):
    to_plot = arr if mask is None else mask
    try:
        n_dims = len(to_plot.dims)
    except AttributeError:
        n_dims = 1

    if n_dims == 2:
        quad = None
        if arr is not None:
            ax, quad = imshow_arr(arr, ax=ax, over=over, **kwargs)
        if mask is not None:
            over = quad if over is None else over
            imshow_mask(mask, ax=ax, over=over, **kwargs)
    if n_dims == 1:
        ax = lineplot_arr(arr, ax=ax, mask=mask, **kwargs)

    return ax


def imshow_mask(mask, ax=None, over=None, cmap=None, **kwargs):
    assert over is not None

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = 'Reds'

    if isinstance(cmap, str):
        cmap = cm.get_cmap(name=cmap)

    cmap.set_bad('k', alpha=0)

    ax.imshow(mask.values, cmap=cmap, interpolation='none', vmax=1, vmin=0,
              origin='lower', extent=over.get_extent(), aspect=ax.get_aspect(), **kwargs)


def imshow_arr(arr, ax=None, over=None, origin='lower', aspect='auto', alpha=None,
               vmin=None, vmax=None, cmap=None, **kwargs):
    """
    Similar to plt.imshow but users different default origin, and sets appropriate
    extent on the plotted data.
    :param arr:
    :param ax:
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()

    x, y = arr.coords[arr.dims[0]].values, arr.coords[arr.dims[1]].values
    extent = [y[0], y[-1], x[0], x[-1]]

    if over is None:
        if alpha is not None:
            if vmin is None:
                vmin = arr.min().item()
            if vmax is None:
                vmax = arr.max().item()
            if cmap is None:
                cmap = 'viridis'
            if isinstance(cmap, str):
                cmap = cm.get_cmap(cmap)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
            mapped_colors = mappable.to_rgba(arr.values)
            mapped_colors[:,:,3] = alpha
            quad = ax.imshow(mapped_colors, origin=origin, extent=extent, aspect=aspect, **kwargs)
        else:
            quad = ax.imshow(arr.values, origin=origin, extent=extent, aspect=aspect, cmap=cmap, **kwargs)
        ax.grid(False)
        ax.set_xlabel(arr.dims[1])
        ax.set_ylabel(arr.dims[0])
    else:
        quad = ax.imshow(arr.values, extent=over.get_extent(), aspect=ax.get_aspect(), origin=origin, **kwargs)

    return ax, quad


def dos_axes(orientation='horiz', figsize=None, with_cbar=True):
    """
    Orientation option should be 'horiz' or 'vert'

    :param orientation:
    :param figsize:
    :param with_cbar:
    :return:
    """
    if figsize is None:
        figsize = (12, 9,) if orientation == 'vert' else (9, 9)

    fig = plt.figure(figsize=figsize)

    outer_grid = gridspec.GridSpec(4, 4, wspace=0.0, hspace=0.0)

    if orientation == 'horiz':
        fig.subplots_adjust(hspace=0.00)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        axes = (ax0, plt.subplot(gs[1], sharex=ax0))
        plt.setp(axes[0].get_xticklabels(), visible=False)
    else:
        fig.subplots_adjust(wspace=0.00)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

        ax0 = plt.subplot(gs[1])
        axes = (ax0, plt.subplot(gs[0], sharey=ax0))
        plt.setp(axes[0].get_yticklabels(), visible=False)

    return fig, axes


def inset_cut_locator(data, reference_data=None, ax=None, location=None, color=None, **kwargs):
    """
    Plots a reference cut location
    :param data:
    :param reference_data:
    :param ax:
    :param location:
    :param kwargs:
    :return:
    """
    quad = data.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    try:
        quad.colorbar.remove()
    except Exception:
        pass

    # add more as necessary
    missing_dim_resolvers = {
        'theta': lambda: reference_data.S.theta,
        'beta': lambda: reference_data.S.beta,
        'phi': lambda: reference_data.S.phi,
    }

    missing_dims = [d for d in data.dims if d not in location]
    missing_values = {d: missing_dim_resolvers[d]() for d in missing_dims}
    ordered_selector = [location.get(d, missing_values.get(d)) for d in data.dims]

    n = 200

    def resolve(name, value):
        if isinstance(value, slice):
            low = value.start
            high = value.stop

            if low is None:
                low = data.coords[name].min().item()
            if high is None:
                high = data.coords[name].max().item()

            return np.linspace(low, high, n)

        return np.ones((n,)) * value

    n_cut_dims = len([d for d in ordered_selector if isinstance(d, (collections.Iterable, slice))])
    ordered_selector = [resolve(d, v) for d, v in zip(data.dims, ordered_selector)]

    if missing_dims:
        assert reference_data is not None
        print(missing_dims)

    if n_cut_dims == 2:
        # a region cut, illustrate with a rect or by suppressing background
        return
    if color is None:
        color = 'red'

    if n_cut_dims == 1:
        # a line cut, illustrate with a line
        ax.plot(*ordered_selector[::-1], color=color, **kwargs)
        pass
    elif n_cut_dims == 0:
        # a single point cut, illustrate with a marker

        pass


def generic_colormap(low, high):
    delta = (high - low)
    low = low - delta / 6
    high = high + delta / 6

    def get_color(value):
        return cm.Blues(float((value - low) / (high - low)))

    return get_color


def phase_angle_colormap(low=0, high=np.pi * 2):
    def get_color(value):
        return cm.twilight_shifted(float((value - low) / (high - low)))

    return get_color


def delay_colormap(low=-1, high=1):
    def get_color(value):
        return cm.coolwarm(float((value - low) / (high - low)))

    return get_color


def temperature_colormap(high=300, low=0, cmap=None):
    if cmap is None:
        cmap = cm.Blues_r

    def get_color(value):
        return cmap(float((value - low) / (high - low)))

    return get_color


def temperature_colormap_around(central, range=50):
    def get_color(value):
        return cm.RdBu_r(float((value - central) / range))

    return get_color


def generic_colorbar(low, high, label='', cmap=None, ax=None, ticks=None, **kwargs):
    extra_kwargs = {
        'orientation': 'horizontal',
        'label': label,
        'ticks': ticks if ticks is not None else [low, high],
    }

    delta = (high - low)
    low = low - delta / 6
    high = high + delta / 6

    extra_kwargs.update(kwargs)
    cb = colorbar.ColorbarBase(
        ax, cmap=cm.get_cmap(cmap or 'Blues'), norm=colors.Normalize(vmin=low, vmax=high), **extra_kwargs)

    return cb


def phase_angle_colorbar(high=np.pi * 2, low=0, ax=None, **kwargs):
    extra_kwargs = {
        'orientation': 'horizontal',
        'label': 'Angle',
        'ticks': ['0', r'$\pi$', r'$2\pi$']
    }

    if not SETTINGS['use_tex']:
        extra_kwargs['ticks'] = ['0', 'π', '2π']

    extra_kwargs.update(kwargs)
    cb = colorbar.ColorbarBase(
        ax, cmap=cm.get_cmap('twilight_shifted'),
        norm=colors.Normalize(vmin=low, vmax=high), **extra_kwargs)
    return cb


def temperature_colorbar(high=300, low=0, ax=None, cmap=None, **kwargs):
    if cmap is None:
        cmap = 'Blues_r'

    extra_kwargs = {
        'orientation': 'horizontal',
        'label': 'Temperature (K)',
        'ticks': [low, high],
    }
    extra_kwargs.update(kwargs)
    cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=colors.Normalize(vmin=low, vmax=high), **extra_kwargs)
    return cb


def delay_colorbar(low=-1, high=1, ax=None, **kwargs):
    # TODO make this nonsequential for use in case where you want to have a long time period after the
    # delay or before
    extra_kwargs = {
        'orientation': 'horizontal',
        'label': 'Probe Pulse Delay (ps)',
        'ticks': [low, 0, high],
    }
    extra_kwargs.update(kwargs)
    cb = colorbar.ColorbarBase(ax, cmap='coolwarm', norm=colors.Normalize(vmin=low, vmax=high), **extra_kwargs)
    return cb


def temperature_colorbar_around(central, range=50, ax=None, **kwargs):
    extra_kwargs = {
        'orientation': 'horizontal',
        'label': 'Temperature (K)',
        'ticks': [central - range, central + range],
    }
    extra_kwargs.update(kwargs)
    cb = colorbar.ColorbarBase(ax, cmap='RdBu_r', norm=colors.Normalize(vmin=central - range, vmax=central + range),
                               **extra_kwargs)
    return cb


colorbarmaps_for_axis = {
    'temp': (temperature_colorbar, temperature_colormap,),
    'delay': (delay_colorbar, delay_colormap,),
    'theta': (phase_angle_colorbar, phase_angle_colormap,),
}


def get_colorbars(fig=None):
    if fig is None:
        fig = plt.gcf()

    colorbars = []
    for ax in fig.axes:
        if ax.get_aspect() == 20:
            colorbars.append(ax)

    return colorbars


def remove_colorbars(fig=None):
    """Removes colorbars from given (or, if no given figure, current) matplotlib figure.

    :param fig (default plt.gcf()):
    """

    # TODO after colorbar removal, plots should be relaxed/rescaled to occupy space previously allocated to colorbars
    # for now, can follow this with plt.tight_layout()
    try:
        if fig is not None:
            for ax in fig.axes:
                if ax.get_aspect() == 20:  # a bit of a hack
                    ax.remove()
        else:
            remove_colorbars(plt.gcf())
    except Exception:
        pass


generic_colorbarmap = (generic_colorbar, generic_colormap,)


def generic_colorbarmap_for_data(data: xr.DataArray, keep_ticks=True, ax=None, **kwargs):
    low, high = data.min().item(), data.max().item()
    ticks = None
    if keep_ticks:
        ticks = data.values
    return (
        generic_colorbar(low=low, high=high, ax=ax, ticks=kwargs.get('ticks', ticks)),
        generic_colormap(low=low, high=high),
    )


def polarization_colorbar(ax=None):
    cb = colorbar.ColorbarBase(ax, cmap='RdBu', norm=colors.Normalize(vmin=-1, vmax=1),
                               orientation='horizontal', label='Polarization', ticks=[-1, 0, 1])
    return cb


def calculate_aspect_ratio(data: DataType):
    data = normalize_to_spectrum(data)

    assert len(data.dims) == 2

    x_extent = np.ptp(data.coords[data.dims[0]].values)
    y_extent = np.ptp(data.coords[data.dims[1]].values)

    return y_extent / x_extent


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """
    Modified from `this StackOverflow question <https://stackoverflow.com/questions/43258638/>`_
    as alternate to the one provided through matplotlib

    size: length of bar in data units
    extent : height of bar ends in axes units
    """

    def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                 label_color=None,
                 frameon=True, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()

        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **kwargs)
        vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **kwargs)
        vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **kwargs)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops={
            'color': label_color,
        })
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                        borderpad=borderpad, child=self.vpac, prop=prop,
                                                        frameon=frameon)


def savefig(desired_path, dpi=400, data=None, **kwargs):
    full_path = path_for_plot(desired_path)
    provenance_path = full_path + '.provenance.json'
    provenance_context = {
        'VERSION': VERSION,
        'time': datetime.datetime.now().isoformat(),
        'jupyter_notebook_name': get_notebook_name(),
        'name': 'savefig',
    }

    def extract(for_data):
        try:
            return for_data.attrs.get('provenance', {})
        except Exception:
            return {}

    if data is not None:
        assert isinstance(data, (list, tuple, set,))
        provenance_context.update({
            'jupyter_context': get_recent_history(1),
            'data': [extract(d) for d in data],
        })
    else:
        # get more recent history because we don't have the data
        provenance_context.update({
            'jupyter_context': get_recent_history(5),
        })

    with open(provenance_path, 'w') as f:
        json.dump(provenance_context, f, indent=2)
    plt.savefig(full_path, dpi=dpi, **kwargs)


def path_for_plot(desired_path):
    workspace = CONFIG['WORKSPACE']

    if not workspace:
        warnings.warn('Saving locally, no workspace found.')
        return os.path.join(os.getcwd(), desired_path)

    if isinstance(workspace, dict):
        workspace = workspace['name']

    try:
        import arpes.config
        filename = os.path.join(arpes.config.FIGURE_PATH, workspace,
                                datetime.date.today().isoformat(), desired_path)
        filename = str(pathlib.Path(filename).absolute())
        parent_directory = os.path.dirname(filename)
        if not os.path.exists(parent_directory):
            try:
                os.makedirs(parent_directory)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise exc

        return filename
    except Exception as e:
        warnings.warn('Misconfigured FIGURE_PATH saving locally: {}'.format(e))
        return os.path.join(os.getcwd(), desired_path)


def path_for_holoviews(desired_path):
    skip_paths = ['.svg', '.png', '.jpeg', '.jpg', '.gif']

    prefix, ext = os.path.splitext(desired_path)

    if ext in skip_paths:
        return prefix

    return prefix + ext


def name_for_dim(dim_name, escaped=True):
    if SETTINGS['use_tex']:
        name = {
            'beta': r'$\beta$',
            'theta': r'$\theta$',
            'chi': r'$\chi$',
            'alpha': r'$\alpha$',
            'psi': r'$\psi$',
            'phi': r'$\phi',
            'eV': r'$\textnormal{E}$',
            'kx': r'$\textnormal{k}_\textnormal{x}$',
            'ky': r'$\textnormal{k}_\textnormal{y}$',
            'kz': r'$\textnormal{k}_\textnormal{z}$',
            'kp': r'$\textnormal{k}_\textnormal{\parallel}$',
            'hv': r'$h\nu$'
        }.get(dim_name)
    else:
        name = {
            'beta': 'β',
            'theta': 'θ',
            'chi': 'χ',
            'alpha': 'α',
            'psi': 'ψ',
            'phi': 'φ',
            'eV': 'E',
            'kx': 'Kx',
            'ky': 'Ky',
            'kz': 'Kz',
            'kp': 'Kp',
            'hv': 'Photon Energy',
        }.get(dim_name)

    if not escaped:
        name = name.replace('$', '')

    return name


def unit_for_dim(dim_name, escaped=True):
    if SETTINGS['use_tex']:
        unit = {
            'theta': r'rad',
            'beta': r'rad',
            'psi': r'rad',
            'chi': r'rad',
            'alpha': r'rad',
            'phi': r'rad',
            'eV': r'eV',
            'kx': r'$\AA^{-1}$',
            'ky': r'$\AA^{-1}$',
            'kz': r'$\AA^{-1}$',
            'kp': r'$\AA^{-1}$',
            'hv': r'eV',
        }.get(dim_name)
    else:
        unit = {
            'theta': r'rad',
            'beta': r'rad',
            'psi': r'rad',
            'chi': r'rad',
            'alpha': r'rad',
            'phi': r'rad',
            'eV': r'eV',
            'kx': '1/Å',
            'ky': '1/Å',
            'kz': '1/Å',
            'kp': '1/Å',
            'hv': 'eV',
        }.get(dim_name)

    if not escaped:
        unit = unit.replace('$', '')

    return unit


def label_for_colorbar(data):
    if not data.S.is_differentiated:
        return r'Spectrum Intensity (arb.)'

    # determine which axis was differentiated
    hist = data.S.history
    records = [h['record'] for h in hist if isinstance(h, dict)]
    if 'curvature' in [r['by'] for r in records]:
        curvature_record = [r for r in records if r['by'] == 'curvature'][0]
        directions = curvature_record['directions']
        return r'Curvature along {} and {}'.format(
            name_for_dim(directions[0]),
            name_for_dim(directions[1])
        )

    derivative_records = [r for r in records if r['by'] == 'dn_along_axis']
    c = Counter(itertools.chain(*[[d['axis']] * d['order'] for d in derivative_records]))

    partial_frag = r''
    if sum(c.values()) > 1:
        partial_frag = r'^' + str(sum(c.values()))

    return r'$\frac{\partial' + partial_frag + r' \textnormal{Int.}}{' + \
           r''.join([r'\partial {}^{}'.format(name_for_dim(item, escaped=False), n)
                     for item, n in c.items()]) + '}$ (arb.)'


def label_for_dim(data=None, dim_name=None, escaped=True):
    if SETTINGS['use_tex']:
        raw_dim_names = {
            'theta': r'$\theta$',
            'beta': r'$\beta$',
            'chi': r'$\chi$',
            'alpha': r'$\alpha$',
            'psi': r'$\psi$',
            'phi': r'$\varphi$',
            'eV': r'Binding Energy (\textbf{eV})',
            'angle': r'Interp. \textbf{Angle}',
            'kinetic': r'Kinetic Energy (\textbf{eV})',
            'temp': r'\textbf{Temperature}',
            'kp': r'$\textbf{k}_\parallel$',
            'kx': r'$\textbf{k}_\text{x}$',
            'ky': r'$\textbf{k}_\text{y}$',
            'kz': r'$\textbf{k}_\perp$',
            'hv': 'Photon Energy',
            'x': 'X (mm)',
            'y': 'Y (mm)',
            'z': 'Z (mm)',
            'spectrum': 'Intensity (arb.)',
        }
    else:
        raw_dim_names = {
            'beta': 'β',
            'theta': 'θ',
            'chi': 'χ',
            'alpha': 'α',
            'psi': 'ψ',
            'phi': 'φ',

            'eV': 'Binding Energy (eV)',
            'angle': 'Interp. Angle',
            'kinetic': 'Kinetic Energy (eV)',
            'temp': 'Temperature (K)',
            'kp': 'Kp',
            'kx': 'Kx',
            'ky': 'Ky',
            'kz': 'Kz',
            'hv': 'Photon Energy (eV)',
            'x': 'X (mm)',
            'y': 'Y (mm)',
            'z': 'Z (mm)',
            'spectrum': 'Intensity (arb.)',
        }

    if dim_name in raw_dim_names:
        return raw_dim_names.get(dim_name)

    try:
        from titlecase import titlecase
    except ImportError:
        warnings.warn('Using alternative titlecase, for better results `pip install titlecase`.')

        def titlecase(s):
            """
            Poor man's titlecase
            :param s:
            :return:
            """
            return s.title()

    return titlecase(dim_name.replace('_', ' '))


def fancy_labels(ax_or_ax_set, data=None):
    """
    Attaches better display axis labels for all axes that can be traversed in the
    passed figure or axes.

    :param ax_or_ax_set:
    :param data:
    :return:
    """
    if isinstance(ax_or_ax_set, (list, tuple, set, np.ndarray)):
        for ax in ax_or_ax_set:
            fancy_labels(ax)
        return

    ax = ax_or_ax_set
    try:
        ax.set_xlabel(label_for_dim(data=data, dim_name=ax.get_xlabel()))
    except Exception:
        pass

    try:
        ax.set_ylabel(label_for_dim(data=data, dim_name=ax.get_ylabel()))
    except Exception:
        pass


def label_for_symmetry_point(point_name):
    if SETTINGS['use_tex']:
        proper_names = {
            'G': r'$\Gamma$',
            'X': r'X',
            'Y': r'Y',
        }
    else:
        proper_names = {
            'G': r'Γ',
            'X': r'X',
            'Y': r'Y',
        }

    return proper_names.get(point_name, point_name)


class CoincidentLinesPlot():
    """
    Helper to allow drawing lines at the same location. Will draw n lines offset so that their
    center appears at the data center, and the lines will end up nonoverlapping.

    Only works for straight lines

    Technique adapted from `StackOverflow
    <https://stackoverflow.com/questions/19394505/matplotlib-expand-the-line-with-specified-width-in-data-unit>`_.
    """

    linewidth = 3

    def __init__(self, **kwargs):
        self.ax = kwargs.pop('ax', plt.gca())
        self.fig = kwargs.pop('fig', plt.gcf())
        self.extra_kwargs = kwargs
        self.ppd = 72. / self.fig.dpi
        self.has_drawn = False

        self.events = {
            'resize_event': self.ax.figure.canvas.mpl_connect('resize_event', self._resize),
            'motion_notify_event': self.ax.figure.canvas.mpl_connect('motion_notify_event', self._resize),
            'button_release_event': self.ax.figure.canvas.mpl_connect('button_release_event', self._resize),
        }
        self.handles = []
        self.lines = []  # saved args and kwargs for plotting, does not verify coincidence

    def add_line(self, *args, **kwargs):
        assert not self.has_drawn
        self.lines.append((args, kwargs,))

    def draw(self):
        self.has_drawn = True

        offset_in_data_units = self.data_units_per_pixel * self.linewidth
        self.offsets = [offset_in_data_units * (o - (len(self.lines) - 1) / 2) for o in range(len(self.lines))]

        for offset, (line_args, line_kwargs) in zip(self.offsets, self.lines):
            line_args = self.normalize_line_args(line_args)
            line_args[1] = np.array(line_args[1]) + offset
            handle = self.ax.plot(*line_args, **line_kwargs)
            self.handles.append(handle)

    @property
    def data_units_per_pixel(self):
        trans = self.ax.transData.transform
        inverse = ((trans((1, 1)) - trans((0, 0))) * self.ppd)
        return (1 / inverse[0], 1 / inverse[1])

    def normalize_line_args(self, args):
        def is_data_type(value):
            return isinstance(value, (
                np.array, np.ndarray, list, tuple
            ))

        assert is_data_type(args[0])

        if len(args) > 1 and is_data_type(args[1]) and len(args[0]) == len(args[1]):
            # looks like we have x and y data
            return args

        # otherwise we should pad the args with the x data
        return [range(len(args[0]))] + args

    def _resize(self, event=None):
        # Keep the trace in here until we can test appropriately.
        import pdb
        pdb.set_trace()
        """
        self.line.set_linewidth(lw)
        self.ax.figure.canvas.draw_idle()
        self.lw = lw
        """


def invisible_axes(ax):
    ax.grid(False)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def no_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
