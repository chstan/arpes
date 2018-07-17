import matplotlib.pyplot as plt
import numpy as np

from arpes.provenance import save_plot_provenance
from arpes.utilities.math import (
    polarization,
    propagate_statistical_error
)
from .utils import *

__all__ = ['spin_polarized_spectrum']


test_polarization = propagate_statistical_error(polarization)


@save_plot_provenance
def spin_polarized_spectrum(spin_dr, title=None, axes=None, out=None, norm=None):
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True)

    ax_left = axes[0]
    ax_right = axes[1]

    up = spin_dr.down.data
    down = spin_dr.up.data

    pol = polarization(spin_dr.up.data, spin_dr.down.data)
    energies = spin_dr.coords['kinetic']
    min_e, max_e = np.min(energies), np.max(energies)

    # Plot the spectra
    ax_left.plot(energies, up, 'r')
    ax_left.plot(energies, down, 'b')
    ax_left.set_title('Spin polarization {}'.format(''))
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
