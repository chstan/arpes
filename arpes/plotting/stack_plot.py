import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import numpy as np

from arpes.analysis import rebin
from arpes.typing import DataType
from arpes.plotting.utils import *
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.plotting.utils import colorbarmaps_for_axis

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


__all__ = ('stack_dispersion_plot', 'flat_stack_plot',)


@save_plot_provenance
def flat_stack_plot(data: DataType, stack_axis=None, fermi_level=True, cbarmap=None, ax=None,
                    title=None, out=None, **kwargs):

    data = normalize_to_spectrum(data)
    if len(data.dims) != 2:
        raise ValueError('In order to produce a stack plot, data must be image-like.'
                         'Passed data included dimensions: {}'.format(data.dims))

    fig = None
    inset_ax = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7, 5,)))
        inset_ax = inset_axes(ax, width='40%', height='5%', loc=1)

    if stack_axis is None:
        stack_axis = data.dims[0]

    skip_colorbar = True
    if cbarmap is None:
        skip_colorbar = False
        cbarmap = colorbarmaps_for_axis[stack_axis]

    cbar, cmap = cbarmap

    # should be exactly two
    other_dim = [d for d in data.dims if d != stack_axis][0]
    other_coord = data.coords[other_dim]

    if not isinstance(cmap, matplotlib.colors.Colormap):
        # do our best
        try:
            cmap = cmap()
        except:
            # might still be fine
            pass

    if 'eV' in data.dims and 'eV' != stack_axis and fermi_level:
        ax.axvline(0, color='red', alpha=0.8, linestyle='--', linewidth=1)

    # meat of the plotting
    for coord_dict, marginal in list(data.T.iterate_axis(stack_axis)):
        marginal.plot(ax=ax, color=cmap(coord_dict[stack_axis]), **kwargs)

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel('Spectrum Intensity (arb).')
    ax.set_title(title, fontsize=14)

    ax.set_xlim([other_coord.min().item(), other_coord.max().item()])

    if inset_ax and not skip_colorbar:
        cbar(ax=inset_ax, **kwargs)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def stack_dispersion_plot(data: DataType, stack_axis=None, ax=None, title=None, out=None,
                          max_stacks=100, transpose=False,
                          use_constant_correction=False, correction_side=None,
                          color=None, c=None,
                          label=None,
                          shift=0,
                          no_scatter=False,
                          negate=False, s=1, scale_factor=None, linewidth=1, palette=None, **kwargs):
    data = normalize_to_spectrum(data)

    if stack_axis is None:
        stack_axis = data.dims[0]

    other_axes = list(data.dims)
    other_axes.remove(stack_axis)
    other_axis = other_axes[0]

    stack_coord = data.coords[stack_axis]
    if len(stack_coord.values) > max_stacks:
        data = rebin(data, reduction=dict([[
            stack_axis, int(np.ceil(len(stack_coord.values) / max_stacks))]
        ]))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if title is None:
        title = '{} Stack'.format(data.S.label.replace('_', ' '))

    max_over_stacks = np.max(data.values)

    cvalues = data.coords[other_axis].values
    if scale_factor is None:
        maximum_deviation = -np.inf

        for _, marginal in data.T.iterate_axis(stack_axis):
            marginal_values = -marginal.values if negate else marginal.values
            marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

            if use_constant_correction:
                true_ys = (marginal_values - marginal_offset)
            else:
                true_ys = (marginal_values - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values)))

            maximum_deviation = np.max([maximum_deviation] + list(np.abs(true_ys)))

        scale_factor = 0.02 * (np.max(cvalues) - np.min(cvalues)) / maximum_deviation

    iteration_order = -1 # might need to fiddle with this in certain cases
    lim = [-np.inf, np.inf]
    labeled = False
    for i, (coord_dict, marginal) in enumerate(list(data.T.iterate_axis(stack_axis))[::iteration_order]):
        coord_value = coord_dict[stack_axis]

        xs = cvalues
        marginal_values = -marginal.values if negate else marginal.values
        marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

        if use_constant_correction:
            offset = right_marginal_offset if correction_side == 'right' else marginal_offset
            true_ys = (marginal_values - offset) / max_over_stacks
            ys = scale_factor * true_ys + coord_value
        else:
            true_ys = (marginal_values - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values))) \
                      / max_over_stacks
            ys = scale_factor * true_ys + coord_value

        colors = color or c or 'black'

        if palette:
            if isinstance(palette, str):
                palette = cm.get_cmap(palette)
            colors = palette(np.abs(true_ys / max_over_stacks))

        if transpose:
            xs, ys = ys, xs

        xs = xs - i * shift

        lim = [max(lim[0], np.min(xs)), min(lim[1], np.max(xs))]

        label_for = '_nolegend_'
        if not labeled:
            labeled = True
            label_for = label

        color_for_plot = colors
        if callable(color_for_plot):
            color_for_plot = color_for_plot(coord_value)

        if isinstance(colors, (str, tuple)) or no_scatter:
            ax.plot(xs, ys, linewidth=linewidth, color=color_for_plot, label=label_for, **kwargs)
        else:
            ax.scatter(xs, ys, color=color_for_plot, s=s, label=label_for, **kwargs)

    x_label = other_axis
    y_label = stack_axis

    if transpose:
        x_label, y_label = y_label, x_label

    ax.set_xlabel(label_for_dim(data, x_label))
    ax.set_ylabel(label_for_dim(data, y_label))

    if transpose:
        ax.set_ylim(lim)
    else:
        ax.set_xlim(lim)

    ax.set_title(title)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def overlapped_stack_dispersion_plot(data: DataType, stack_axis=None, ax=None, title=None, out=None,
                                     max_stacks=100, use_constant_correction=False, transpose=False,
                                     negate=False, s=1, scale_factor=None, linewidth=1, palette=None, **kwargs):
    data = normalize_to_spectrum(data)

    if stack_axis is None:
        stack_axis = data.dims[0]

    other_axes = list(data.dims)
    other_axes.remove(stack_axis)
    other_axis = other_axes[0]

    stack_coord = data.coords[stack_axis]
    if len(stack_coord.values) > max_stacks:
        data = rebin(data, reduction=dict([[
            stack_axis, int(np.ceil(len(stack_coord.values) / max_stacks))]
        ]))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if title is None:
        title = '{} Stack'.format(data.S.label.replace('_', ' '))

    max_over_stacks = np.max(data.values)

    cvalues = data.coords[other_axis].values
    if scale_factor is None:
        maximum_deviation = -np.inf

        for _, marginal in data.T.iterate_axis(stack_axis):
            marginal_values = -marginal.values if negate else marginal.values
            marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

            if use_constant_correction:
                true_ys = (marginal_values - marginal_offset)
            else:
                true_ys = (marginal_values - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values)))

            maximum_deviation = np.max([maximum_deviation] + list(np.abs(true_ys)))

        scale_factor = 0.02 * (np.max(cvalues) - np.min(cvalues)) / maximum_deviation

    iteration_order = -1 # might need to fiddle with this in certain cases
    for coord_dict, marginal in list(data.T.iterate_axis(stack_axis))[::iteration_order]:
        coord_value = coord_dict[stack_axis]

        xs = cvalues
        marginal_values = -marginal.values if negate else marginal.values
        marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

        if use_constant_correction:
            true_ys = (marginal_values - marginal_offset) / max_over_stacks
            ys = scale_factor * true_ys + coord_value
        else:
            true_ys = (marginal_values - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values))) \
                      / max_over_stacks
            ys = scale_factor * true_ys + coord_value

        colors = 'black'
        if palette:
            if isinstance(palette, str):
                palette = cm.get_cmap(palette)
            colors = palette(np.abs(true_ys / max_over_stacks))

        if transpose:
            xs, ys = ys, xs

        if isinstance(colors, str):
            plt.plot(xs, ys, linewidth=linewidth, color=colors, **kwargs)
        else:
            plt.scatter(xs, ys, color=colors, s=s, **kwargs)

    x_label = other_axis
    y_label = stack_axis

    if transpose:
        x_label, y_label = y_label, x_label

    ax.set_xlabel(label_for_dim(data, x_label))
    ax.set_ylabel(label_for_dim(data, y_label))

    ax.set_title(title)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()

    return fig, ax
