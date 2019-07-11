"""
This module is a bit of a misnomer, in that it also applies perfectly well to data collected by a delay
line on a hemisphere, the important point is that the data in any given channel should correspond to the true number of
electrons that arrived in that channel.

Plotting routines here are ones that include statistical errorbars. Generally for datasets in PyARPES, an xr.Dataset
will hold the standard deviation data for a given variable on `{var_name}_std`.
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from arpes.provenance import save_plot_provenance
from arpes.typing import DataType
from arpes.plotting.utils import *

__all__ = ('plot_with_std', 'scatter_with_std',)

@save_plot_provenance
def jitterplot(data: xr.Dataset, jitter_dimension=None):
    if jitter_dimension is None:
        jitter_dimension = 'bootstrap'

    assert(jitter_dimension in data.coords)



@save_plot_provenance
def plot_with_std(data: DataType, name_to_plot=None, ax=None, out=None, **kwargs):
    if name_to_plot is None:
        var_names = [k for k in data.data_vars.keys() if '_std' not in k]
        assert (len(var_names) == 1)
        name_to_plot = var_names[0]
        assert ((name_to_plot + '_std') in data.data_vars.keys())

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (7, 5,)))

    data.data_vars[name_to_plot].plot(ax=ax, **kwargs)
    x, y = data.data_vars[name_to_plot].T.to_arrays()

    std = data.data_vars[name_to_plot + '_std'].values
    ax.fill_between(x, y - std, y + std, alpha=0.3, **kwargs)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    ax.set_xlim([np.min(x), np.max(x)])

    return fig, ax


@save_plot_provenance
def scatter_with_std(data: DataType, name_to_plot=None, ax=None, fmt='o', out=None, **kwargs):
    if name_to_plot is None:
        var_names = [k for k in data.data_vars.keys() if '_std' not in k]
        assert(len(var_names) == 1)
        name_to_plot = var_names[0]
        assert((name_to_plot + '_std') in data.data_vars.keys())

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (7, 5,)))

    x, y = data.data_vars[name_to_plot].T.to_arrays()

    std = data.data_vars[name_to_plot + '_std'].values
    ax.errorbar(x, y, yerr=std, fmt=fmt, markeredgecolor='black', **kwargs)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    ax.set_xlim([np.min(x), np.max(x)])

    return fig, ax