import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors, gridspec

from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.analysis.xps import approximate_core_levels
from .utils import *

__all__ = ('plot_dos', 'plot_core_levels',)


@save_plot_provenance
def plot_core_levels(data, title=None, out=None, norm=None, dos_pow=1, core_levels=None, binning=1, promenance=5, **kwargs):
    axes, cbar = plot_dos(data=data, title=title, out=None, norm=norm, dos_pow=dos_pow, **kwargs)

    if core_levels is None:
        core_levels = approximate_core_levels(data, binning=binning, promenance=promenance)

    for core_level in core_levels:
        axes[1].axvline(core_level, color='red', ls='--')

    if out is not None:
        savefig(out, dpi=400)
        return path_for_plot(out)
    else:
        return axes, cbar


@save_plot_provenance
def plot_dos(data, title=None, out=None, norm=None, dos_pow=1, **kwargs):
    data = normalize_to_spectrum(data)

    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(hspace=0.00)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    axes = (ax0, plt.subplot(gs[1], sharex=ax0))

    data.values[np.isnan(data.values)] = 0
    cbar_axes = matplotlib.colorbar.make_axes(axes, pad=0.01)

    mesh = data.plot(ax=axes[0], norm=norm or colors.PowerNorm(gamma=0.15))

    axes[1].set_facecolor((0.95, 0.95, 0.95))
    density_of_states = data.S.sum_other(['eV'])
    (density_of_states ** dos_pow).plot(ax=axes[1])

    cbar = plt.colorbar(mesh, cax=cbar_axes[0])
    cbar.set_label('Photoemission Intensity (Arb.)')

    axes[1].set_ylabel('Spectrum DOS', labelpad=12)
    axes[0].set_title(title or '')

    if out is not None:
        savefig(out, dpi=400)
        return path_for_plot(out)
    else:
        return fig, axes, cbar
