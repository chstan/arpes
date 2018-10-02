from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward
from arpes.plotting.utils import unit_for_dim, name_for_dim
import numpy as np

__all__ = ('annotate_cuts', 'annotate_point',)

def annotate_cuts(ax, data, plotted_axes, include_text_labels=False, **kwargs):
    """
    Example: annotate_cuts(ax, conv, ['kz', 'ky'], hv=80)

    :param ax:
    :param data:
    :param plotted_axes:
    :param include_text_labels:
    :param kwargs:
    :return:
    """
    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    assert (len(plotted_axes) == 2)

    for k, v in kwargs.items():
        if not isinstance(v, (tuple, list, np.ndarray,)):
            v = [v]

        selected = converted_coordinates.sel(**dict([[k, v]]), method='nearest')

        for coords_dict, obj in selected.T.iterate_axis(k):
            css = [obj[d].values for d in plotted_axes]
            ax.plot(*css, color='red', ls='--', linewidth=1, dashes=(5, 5))

            if include_text_labels:
                idx = np.argmin(css[1])

                ax.text(css[0][idx] + 0.05, css[1][idx], '{} = {} {}'.format(name_for_dim(k), coords_dict[k].item(), unit_for_dim(k)),
                        color='red', size='medium')


def annotate_point(ax, location, label, delta=None, **kwargs):
    label = {
        'G': '$\\Gamma$',
        'X': r'\textbf{X}',
        'Y': r'\textbf{Y}',
        'K': r'\textbf{K}',
        'M': r'\textbf{M}',
    }.get(label, label)

    if delta is None:
        delta = (-0.05, 0.05,)

    if len(delta) == 2:
        dx, dy = tuple(delta)
        x, y = tuple(location)
        ax.plot([x], [y], 'o', c='red')
        ax.text(x + dx, y + dy, label, color='red', **kwargs)
    else:
        dx, dy, dz = tuple(delta)
        x, y, z = tuple(location)
        ax.plot([x], [y], [z], 'o', c='red')
        ax.text(x + dx, y + dy, z + dz, label, color='red', **kwargs)