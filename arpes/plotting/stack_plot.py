import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from analysis import rebin
from arpes.typing import DataType
from plotting.utils import *
from provenance import save_plot_provenance
from utilities import normalize_to_spectrum

__all__ = ('stack_dispersion_plot',)

@save_plot_provenance
def stack_dispersion_plot(data: DataType, stack_axis=None, ax=None, title=None, out=None,
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
