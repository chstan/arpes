from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward
from arpes.plotting.utils import unit_for_dim, name_for_dim
import numpy as np

__all__ = ('annotate_cuts',)

def annotate_cuts(ax, data, plotted_axes, include_text_labels=False, **kwargs):
    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    assert (len(plotted_axes) == 2)

    for k, v in kwargs.items():
        selected = converted_coordinates.sel(**dict([[k, v]]), method='nearest')

        for coords_dict, obj in selected.T.iterate_axis(k):
            css = [obj[d].values for d in plotted_axes]
            ax.plot(*css, color='red', ls='--')

            if include_text_labels:
                idx = np.argmin(css[1])

                ax.text(css[0][idx] + 0.05, css[1][idx], '{} = {} {}'.format(name_for_dim(k), coords_dict[k].item(), unit_for_dim(k)),
                        color='red', size='medium')
