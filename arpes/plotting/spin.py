import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection

from arpes.analysis.sarpes import to_intensity_polarization
from arpes.provenance import save_plot_provenance
from arpes.utilities.math import (
    polarization,
    propagate_statistical_error
)
from .utils import *

__all__ = ('spin_polarized_spectrum', 'spin_colored_spectrum',
           'spin_difference_spectrum',)


test_polarization = propagate_statistical_error(polarization)

@save_plot_provenance
def spin_colored_spectrum(spin_dr, title=None, axes=None, out=None, scatter=False, **kwargs):
    if axes is None:
        _, axes = plt.subplots(figsize=(6, 4))

    as_intensity = to_intensity_polarization(spin_dr)
    intensity = as_intensity.intensity
    pol = as_intensity.polarization.copy(deep=True)

    if len(intensity.dims) == 1:
        inset_ax = inset_axes(axes, width="30%", height="5%", loc=1)
        coord = intensity.coords[intensity.dims[0]]
        points = np.array([coord.values, intensity.values]).T.reshape(-1, 1, 2)
        pol.values[np.isnan(pol.values)] = 0
        pol.values[pol.values > 1] = 1
        pol.values[pol.values < -1] = -1
        colors = cm.get_cmap('RdBu')(pol.values[:-1])

        if scatter:
            colors = cm.get_cmap('RdBu')(pol.values)
            axes.scatter(coord.values, intensity.values, c=colors, s=1.5)
        else:
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors)

            axes.add_collection(lc)

        axes.set_xlim(coord.min().item(), coord.max().item())
        axes.set_ylim(0, intensity.max().item() * 1.15)
        axes.set_ylabel('ARPES Spectrum Intensity (arb.)')
        axes.set_xlabel(label_for_dim(spin_dr, dim_name=intensity.dims[0]))
        axes.set_title(title if title is not None else 'Spin Polarization')
        polarization_colorbar(inset_ax)


    if out is not None:
        savefig(out, dpi=400)
        plt.clf()
        return path_for_plot(out)
    else:
        plt.show()

@save_plot_provenance
def spin_difference_spectrum(spin_dr, title=None, axes=None, out=None, scatter=False, **kwargs):
    if axes is None:
        _, axes = plt.subplots(figsize=(6, 4))

    as_intensity = to_intensity_polarization(spin_dr)
    intensity = as_intensity.intensity
    pol = as_intensity.polarization.copy(deep=True)

    if len(intensity.dims) == 1:
        inset_ax = inset_axes(axes, width="30%", height="5%", loc=1)
        coord = intensity.coords[intensity.dims[0]]
        points = np.array([coord.values, intensity.values]).T.reshape(-1, 1, 2)
        pol.values[np.isnan(pol.values)] = 0
        pol.values[pol.values > 1] = 1
        pol.values[pol.values < -1] = -1
        colors = cm.get_cmap('RdBu')(pol.values[:-1])

        if scatter:
            colors = cm.get_cmap('RdBu')(pol.values)
            axes.scatter(coord.values, intensity.values, c=colors, s=1.5)
        else:
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors)

            axes.add_collection(lc)

        axes.set_xlim(coord.min().item(), coord.max().item())
        axes.set_ylim(0, intensity.max().item() * 1.15)
        axes.set_ylabel('ARPES Spectrum Intensity (arb.)')
        axes.set_xlabel(label_for_dim(spin_dr, dim_name=intensity.dims[0]))
        axes.set_title(title if title is not None else 'Spin Polarization')
        polarization_colorbar(inset_ax)

    if out is not None:
        savefig(out, dpi=400)
        plt.clf()
        return path_for_plot(out)
    else:
        plt.show()


@save_plot_provenance
def spin_polarized_spectrum(spin_dr, title=None, axes=None, out=None, norm=None):
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True)

    ax_left = axes[0]
    ax_right = axes[1]

    up = spin_dr.down.data
    down = spin_dr.up.data

    pol = polarization(spin_dr.up.data, spin_dr.down.data)
    energies = spin_dr.coords['eV']
    min_e, max_e = np.min(energies), np.max(energies)

    # Plot the spectra
    ax_left.plot(energies, up, 'r')
    ax_left.plot(energies, down, 'b')
    ax_left.set_title(title if title is not None else 'Spin polarization {}'.format(''))
    ax_left.set_ylabel(r'\textbf{Spectrum Intensity}')
    ax_left.set_xlabel(r'\textbf{Kinetic energy} (eV)')
    ax_left.set_xlim(min_e, max_e)

    max_up = np.max(up)
    max_down = np.max(down)
    plt.ylim(0, max(max_down, max_up) * 0.7)

    # Plot the polarization and associated statistical error bars
    ax_right.plot(energies, pol, color='black')
    ax_right.fill_between(energies, 0, 1, facecolor='red', alpha=0.1)
    ax_right.fill_between(energies, -1, 0, facecolor='blue', alpha=0.1)
    #ax_right.fill_between(energies, pol - 3 * (test_pol + 0.005),
    #                      pol + 3 * (test_pol + 0.005), facecolor='black', alpha=0.3)
    ax_right.set_title('Spin polarization')
    ax_right.set_ylabel(r'\textbf{Polarization}')
    ax_right.set_xlabel(r'\textbf{Kinetic Energy} (eV)')
    ax_right.set_xlim(min_e, max_e)
    ax_right.axhline(0, color='white', linestyle=':')

    ax_right.set_ylim(-1, 1)
    ax_right.grid(True, axis='y')

    plt.tight_layout()

    if out is not None:
        savefig(out, dpi=400)
        plt.clf()
        return path_for_plot(out)
    else:
        plt.show()
