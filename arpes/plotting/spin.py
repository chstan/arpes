import matplotlib.pyplot as plt
import numpy

from arpes.utilities.math import (
    polarization,
    propagate_statistical_error
)

test_polarization = propagate_statistical_error(polarization)

def prettyplot():
    pass

def spin_polarized_spectrum(spin_edc):
    up = numpy.copy(spin_edc.spin_up_energy)
    down = numpy.copy(spin_edc.spin_down_energy)

    test_pol = test_polarization(up, down)
    pol = polarization(up, down)

    # Plot the spectra
    spectra_fig = plt.subplot(2, 1, 1)
    plt.plot(spin_edc.energies, spin_edc.spin_up_energy, 'r')
    plt.plot(spin_edc.energies, spin_edc.spin_down_energy, 'b')
    plt.title('Spin up counts')
    plt.ylabel('Counts')
    plt.xlabel('Electron kinetic energy')
    plt.xlim(5.5, 7)

    max_up = numpy.max(up)
    max_down = numpy.max(down)
    plt.ylim(0, max(max_down, max_up) * 0.7)

    # Plot the polarization and associated statistical error bars
    pol_fig = plt.subplot(2, 1, 2)
    plt.plot(spin_edc.energies, pol, color='black')
    plt.fill_between(spin_edc.energies, 0, 1, facecolor='red', alpha=0.1)
    plt.fill_between(spin_edc.energies, -1, 0, facecolor='blue', alpha=0.1)
    plt.fill_between(spin_edc.energies, pol - 3 * (test_pol + 0.005), pol + 3 * (test_pol + 0.005), facecolor='black', alpha=0.3)
    plt.title('Polarization in spin resolved spectra')
    plt.ylabel('Polarization')
    plt.xlabel('Electron kinetic energy')
    plt.xlim(5.5, 7)
    plt.ylim(-0.2, 0.2)

    pol_fig.grid(True, axis='y')
