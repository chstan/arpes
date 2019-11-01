import matplotlib.pyplot as plt
import numpy as np

from arpes.plotting.utils import simple_ax_grid

__all__ = ('plot_fit', 'plot_fits',)


def plot_fit(model_result, ax=None):
    """
    Performs a straightforward plot of the data, residual, and fit to an axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = model_result.userkws[model_result.model.independent_vars[0]]
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.axhline(0, color='green', linestyle='--', alpha=0.5)

    ax.scatter(x, model_result.data, s=10, edgecolors='blue', marker='s', c='white', linewidth=1.5)
    ax.plot(x, model_result.best_fit, color='red', linewidth=1.5)

    ax2.scatter(x, model_result.residual, edgecolors='green', alpha=0.5, s=12, marker='s', c='white', linewidth=1.5)
    ylim = np.max(np.abs(np.asarray(ax2.get_ylim()))) * 1.5
    ax2.set_ylim([-ylim,ylim])
    ax.set_xlim([np.min(x), np.max(x)])


def plot_fits(model_results, ax=None):
    n_results = len(model_results)
    if ax is None:
        fig, ax, ax_extra = simple_ax_grid(n_results, sharex='col', sharey='row')

    for axi, model_result in zip(ax, model_results):
        plot_fit(model_result, ax=axi)