import warnings

import math
import matplotlib.pyplot as plt
import numpy as np

from arpes.provenance import save_plot_provenance
from .utils import *

from arpes.fits import broadcast_model, GStepBModel
from arpes.utilities import apply_dataarray

__all__ = ['fermi_edge_reference', 'plot_fit']

@save_plot_provenance
def plot_fit(data, title=None, axes=None, out=None, norm=None, **kwargs):
    """
    Plots the results of a fit of some lmfit model to some data.

    We introspect the model to determine which attributes we should plot,
    as well as their uncertainties
    :param data: The data, this should be of type DataArray<lmfit.model.ModelResult>
    :param title:
    :param ax:
    :param out:
    :param norm:
    :param kwargs:
    :return:
    """

    # get any of the elements
    reference_fit = data.values.ravel()[0]
    model = reference_fit.model
    SKIP_NAMES = {'const_bkg', 'lin_bkg', 'erf_amp'}
    param_names = [p for p in model.param_names if p not in SKIP_NAMES]
    n_params = len(param_names)
    MAX_COLS = 3
    n_rows = int(math.ceil(n_params / MAX_COLS))
    n_cols = n_params if n_params < MAX_COLS else MAX_COLS

    is_bootstrapped = 'bootstrap' in data.dims

    if axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    for i, param in enumerate(param_names):
        row = i // MAX_COLS
        column = i - (row * MAX_COLS)

        try:
            ax = axes[row, column]
        except IndexError:
            ax = axes[column] # n_rows = 1

        # extract the data for this param name
        # attributes are on .value and .stderr
        centers = data.T.map(lambda x: x.params[param].value)
        if is_bootstrapped:
            centers = centers.mean('bootstrap')

        centers.plot(ax=ax)

        ax.set_title('Fit var: {}'.format(param))

        if len(centers.dims) == 1:
            if is_bootstrapped:
                widths = data.T.map(lambda x: x.params[param].value).std('bootstrap')
            else:
                widths = data.T.map(lambda x: x.params[param].stderr)
            # then we can plot widths as well, otherwise we need more
            # figures, blergh
            x_coords = centers.coords[centers.dims[0]]
            ax.fill_between(x_coords, centers.values + widths.values,
                            centers.values - widths.values, alpha=0.5)


    if title is None:
        title = data.S.label.replace('_', ' ')

    # if multidimensional, we can share y axis as well
    #axes.set_xlabel(label_for_dim(data, axes[0][0].get_xlabel()))
    #ax.set_title(title, font_size=14)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()


@save_plot_provenance
def fermi_edge_reference(data, title=None, ax=None, out=None, norm=None, **kwargs):
    warnings.warn('Not automatically correcting for slit shape distortions to the Fermi edge')

    sum_dimensions = {'cycle', 'phi', 'kp', 'kx'}
    sum_dimensions.intersection_update(set(data.dims))
    summed_data = data.sum(*list(sum_dimensions))

    broadcast_dimensions = summed_data.dims
    broadcast_dimensions.remove('eV')
    if len(broadcast_dimensions) == 1:
        edge_fit = broadcast_model(GStepBModel, summed_data.sel(eV=slice(-0.1, 0.1)), broadcast_dimensions[0])
    else:
        warnings.warn('Could not product fermi edge reference. Too many dimensions: {}'.format(broadcast_dimensions))
        return

    centers = apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].value, otypes=[np.float]))
    widths = apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['width'].value, otypes=[np.float]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if title is None:
        title = data.S.label.replace('_', ' ')

    plot_centers = centers.plot(norm=norm, ax=ax)
    plot_widths = widths.plot(norm=norm, ax=ax)

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))

    ax.set_title(title, font_size=14)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()
