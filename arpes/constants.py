"""Useful constants for experiments and conversions.

Much of this is collected from past students, especially Jeff's 'Cstes.ipf'.

Some of this will disappear in future updates, as we move away from magic constants towards
bundling necessary information on endstation classes.
"""

from numpy import pi

# eV, A reasonablish value if you aren't sure for the particular sample
WORK_FUNCTION = 4.3

METERS_PER_SECOND_PER_EV_ANGSTROM = (
    151927  # converts from eV * angstrom to meters/second velocity units
)
HBAR = 1.0545718176461565 * 10 ** (-34)
HBAR_PER_EV = 6.582119569 * 10 ** (
    -16
)  # gives the energy lifetime relationship via tau = -hbar / np.imag(self_energy)
BARE_ELECTRON_MASS = 9.109383701e-31  # kg
HBAR_SQ_EV_PER_ELECTRON_MASS = 0.475600805657  # hbar^2 / m0 in eV^2 s^2 / kg
HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ = 7.619964  # (hbar^2) / (m0 * angstrom ^2) in eV

K_BOLTZMANN_EV_KELVIN = 8.617333262145178e-5  # in units of eV / Kelvin
K_BOLTZMANN_MEV_KELVIN = 1000 * K_BOLTZMANN_EV_KELVIN  # meV / Kelvin

HC = 1239.84172  # in units of eV * nm

HEX_ALPHABET = "ABCDEF0123456789"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Lanzara lab specific
STRAIGHT_TOF_LENGTH = 0.937206
SPIN_TOF_LENGTH = 1.1456
DLD_LENGTH = 1.1456  # This isn't correct but it should be a reasonable guess

K_INV_ANGSTROM = 0.5123167219534328
HV_CONVERSION = 3.814697265625

# TODO these should be migrated into their appropriate loaders
SPECTROMETER_MC = {
    "name": "MC",
    "rad_per_pixel": (1 / 10) * (pi / 180),
    "type": "hemisphere",
    "is_slit_vertical": False,
}

SPECTROMETER_MC_OLD = {
    "name": "MC_OLD",
    "type": "hemisphere",
    "rad_per_pixel": 0.125 * (pi / 180),
    "is_slit_vertical": False,
}

SPECTROMETER_STRAIGHT_TOF = {
    "name": "STRAIGHT_ToF",
    "length": STRAIGHT_TOF_LENGTH,
    "mstar": 1.0,
    "type": "tof",
    "dof": ["t"],
    "scan_dof": ["theta"],
}

SPECTROMETER_SPIN_TOF = {
    "name": "SPIN_ToF",
    "length": SPIN_TOF_LENGTH,
    "mstar": 0.5,
    "type": "tof",
    "dof": ["time", "spin"],
    "scan_dof": ["theta", "beta"],
}

SPECTROMETER_DLD = {
    "name": "DLD",
    "length": DLD_LENGTH,
    "type": "tof",
    "dof_type": {
        "timing": ["x_pixels", "t_pixels"],
        "spatial": ["x_pixels", "y_pixels"],
    },
    "scan_dof": ["theta"],
}

SPECTROMETER_BL4 = {
    "name": "BL4",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}

SPECTROMETER_BL7 = {
    "name": "BL7",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}

SPECTROMETER_ANTARES = {
    "name": "ANTARES",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}

SPECTROMETER_KAINDL = {
    "name": "Kaindl",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}
