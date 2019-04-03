from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward
from arpes.plotting.utils import unit_for_dim, name_for_dim
import numpy as np

__all__ = ('annotate_cuts', 'annotate_point', 'annotate_experimental_conditions',)

def annotate_experimental_conditions(ax, data, desc, show=False, orientation='top', **kwargs):
    """
    Renders information about the experimental conditions onto a set of axes,
    also adjust the axes limits and hides the axes.

    data should be the dataset described, and desc should be one of

    'temp',
    'photon',
    'photon polarization',
    'polarization',
    or a number to act as a spacer in units of the axis coordinates

    or a list of such items

    :param ax:
    :param data:
    :param desc:
    :return:
    """

    if isinstance(desc, (str, int, float,)):
        desc = [desc]

    ax.grid(False)
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 100])
    if not show:
        ax.set_axis_off()
        ax.patch.set_alpha(0)

    delta = -1
    current = 100
    if orientation == 'bottom':
        delta = 1
        current = 0

    fontsize = kwargs.pop('fontsize', 16)
    delta = fontsize * delta

    conditions = data.S.experimental_conditions

    def render_polarization(c):
        pol = c['polarization']
        if pol in ['lc', 'rc']:
            return '\\textbf{' + pol.upper() + '}'

        symbol_pol = {
            's': '',
            'p': '',
            's-p': '',
            'p-s': '',
        }

        prefix = ''
        if pol in ['s-p', 'p-s']:
            prefix = '\\textbf{Linear Dichroism, }'

        symbol = symbol_pol[pol]
        if len(symbol):
            return prefix + '$' + symbol + '$/\\textbf{' + pol + '}'

        return prefix + '\\textbf{' + pol + '}'

    def render_photon(c):
        return '\\textbf{' + str(c['hv']) + ' eV'

    renderers = {
        'temp': lambda c: '\\textbf{T = ' + '{:.3g}'.format(c['temp']) + ' K}',
        'photon': render_photon,
        'photon polarization': lambda c: render_photon(c) + ', ' + render_polarization(c),
        'polarization': render_polarization,
    }

    for item in desc:
        if isinstance(item, (float, int)):
            current += item + delta
            continue

        item = item.replace('_', ' ').lower()

        ax.text(0, current, renderers[item](conditions), fontsize=fontsize, **kwargs)
        current += delta


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

    c = kwargs.pop('color', 'red')

    if len(delta) == 2:
        dx, dy = tuple(delta)
        x, y = tuple(location)
        ax.plot([x], [y], 'o', c=c)
        ax.text(x + dx, y + dy, label, color=c, **kwargs)
    else:
        dx, dy, dz = tuple(delta)
        x, y, z = tuple(location)
        ax.plot([x], [y], [z], 'o', c=c)
        ax.text(x + dx, y + dy, z + dz, label, color=c, **kwargs)