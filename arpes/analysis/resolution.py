"""
Contains calibrations and information for spectrometer resolution.
"""
import math
import numpy as np

from arpes.typing import DataType

# all resolutions are given by (photon energy, entrance slit, exit slit size)
from arpes.constants import K_BOLTZMANN_MEV_KELVIN
from arpes.utilities import normalize_to_spectrum

__all__ = ('total_resolution_estimate',)

# all analyzer dimensions are given in millimeters for convenience as this
# is how slit sizes are typically reported
def r8000(slits):
    return {
        'type': 'HEMISPHERE',
        'slits': slits,
        'radius': 200,
        'angle_resolution': 0.1 * np.pi / 180,
    }

def analyzer_resolution(analyzer_information, slit_width=None, slit_number=None,
                        pass_energy=10):
    if slit_width is None:
        slit_width = analyzer_information['slits'][slit_number]

    return 1000 * pass_energy * (slit_width / (2 * analyzer_information['radius']))

SPECTROMETER_INFORMATION = {
    'BL403':  r8000([0.05, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.8])
}

MERLIN_BEAMLINE_RESOLUTION = {
    'LEG': {
        # 40 um by 40 um slits
        (25, (40, 40)): 9.5,
        (30, (40, 40)): 13.5,
        (35, (40, 40)): 22.4,
    },
    'HEG': {
        # 30 um by 30 um slits
        (30, (30, 30)): 5.2,
        (35, (30, 30)): 5.4,
        (40, (30, 30)): 5.0,
        (45, (30, 30)): 5.5,
        (50, (30, 30)): 5.4,
        (55, (30, 30)): 5.5,
        (60, (30, 30)): 6.4,
        (65, (30, 30)): 6.5,
        (70, (30, 30)): 7.7,
        (75, (30, 30)): 9,
        (80, (30, 30)): 8.6,
        (90, (30, 30)): 12.4,
        (100, (30, 30)): 21.6,
        (110, (30, 30)): 35.4,

        # 50um by 50um
        (60, (50, 50)): 8.2,
        (70, (50, 50)): 10.2,
        (80, (50, 50)): 12.6,
        (90, (50, 50)): 16.5,

        # 60um by 80um
        (60, (60, 80)): 9.2,
        (70, (60, 80)): 13.0,
        (80, (60, 80)): 16.5,
        (90, (60, 80)): 22.0,

        # 90um by 140um
        (30, (90, 140)): 7.3,
        (40, (90, 140)): 9,
        (50, (90, 140)): 11.9,
        (60, (90, 140)): 16.2,
        (70, (90, 140)): 21.4,
        (80, (90, 140)): 25.8,
        (90, (90, 140)): 34,
        (100, (90, 140)): 45,
        (110, (90, 140)): 60,
    },
    # second order from the high density grating
    'HEG-2': {
        # 30um by 30um
        (90, (30, 30)): 8,
        (100, (30, 30)): 9,
        (110, (30, 30)): 9.6,
        (120, (30, 30)): 9.6,
        (130, (30, 30)): 12,
        (140, (30, 30)): 15,
        (150, (30, 30)): 13,

        # 50um by 50um
        (90, (50, 50)): 10.3,
        (100, (50, 50)): 10.5,
        (110, (50, 50)): 13.2,
        (120, (50, 50)): 14,
        (130, (50, 50)): 19,
        (140, (50, 50)): 22,
        (150, (50, 50)): 22,

        # 60um by 80um
        (90, (60, 80)): 12.8,
        (100, (60, 80)): 15,
        (110, (60, 80)): 16.4,
        (120, (60, 80)): 19,
        (130, (60, 80)): 27,
        (140, (60, 80)): 30,

        # 90um by 140um
        (90, (90, 140)): 19,
        (100, (90, 140)): 21,
        (110, (90, 140)): 28,
        (120, (90, 140)): 31,
        (130, (90, 140)): 37,
        (140, (90, 140)): 41,
        (150, (90, 140)): 49,
    },
}

ENDSTATIONS_BEAMLINE_RESOLUTION = {
    'BL403': MERLIN_BEAMLINE_RESOLUTION,
}


def analyzer_resolution_estimate(data: DataType, meV=False):
    """
    For hemispherical analyzers, this can be determined by the slit
    and pass energy settings.

    Roughly,
    :param data:
    :return:
    """
    data = normalize_to_spectrum(data)

    endstation = data.S.endstation
    spectrometer_info = SPECTROMETER_INFORMATION[endstation]

    spectrometer_settings = data.S.spectrometer_settings

    return analyzer_resolution(spectrometer_info, slit_number=spectrometer_settings['slit'],
                               pass_energy=spectrometer_settings['pass_energy']) * (1 if meV else 0.001)


def energy_resolution_from_beamline_slit(table, photon_energy, exit_slit_size):
    """
    Assumes an exact match on the photon energy, though that interpolation
    could also be pulled into here...
    :param table:
    :param photon_energy:
    :param exit_slit_size:
    :return:
    """

    by_slits = {k[1]: v for k, v in table.items() if k[0] == photon_energy}
    if exit_slit_size in by_slits:
        return by_slits[exit_slit_size]

    slit_area = exit_slit_size[0] * exit_slit_size[1]
    by_area = {int(k[0] * k[1]): v for k, v in by_slits.items()}

    if len(by_area) == 1:
        return list(by_area.values())[0] * slit_area / (list(by_area.keys())[0])

    try:
        low = max(k for k in by_area.keys() if k <= slit_area)
        high = min(k for k in by_area.keys() if k >= slit_area)
    except ValueError:
        if slit_area > max(by_area.keys()):
            # use the largest and second largest
            high = max(by_area.keys())
            low = max(k for k in by_area.keys() if k < high)
        else:
            # use the smallest and second smallest
            low = min(by_area.keys())
            high = min(k for k in by_area.keys() if k > low)

    return by_area[low] + (by_area[high] - by_area[low]) * (slit_area - low) / (high - low)


def beamline_resolution_estimate(data: DataType, meV=False):
    data = normalize_to_spectrum(data)
    resolution_table = ENDSTATIONS_BEAMLINE_RESOLUTION[data.S.endstation]

    if isinstance(list(resolution_table.keys())[0], str):
        # need grating information
        settings = data.S.beamline_settings
        resolution_table = resolution_table[settings['grating']]

        all_keys = list(resolution_table.keys())
        hvs = set(k[0] for k in all_keys)

        low_hv = max(hv for hv in hvs if hv < settings['hv'])
        high_hv = min(hv for hv in hvs if hv >= settings['hv'])

        slit_size = (settings['entrance_slit'], settings['exit_slit'],)
        low_hv_res = energy_resolution_from_beamline_slit(
            resolution_table, low_hv, slit_size)
        high_hv_res = energy_resolution_from_beamline_slit(
            resolution_table, high_hv, slit_size)

        # interpolate between nearest values
        return low_hv_res + (high_hv_res - low_hv_res) * \
                            (settings['hv'] - low_hv) / (high_hv - low_hv) * (1000 if meV else 1)

    raise NotImplementedError()


def thermal_broadening_estimate(data: DataType, meV=False):
    """
    Simple Fermi-Dirac broadening
    :param data:
    :return:
    """
    return normalize_to_spectrum(data).S.temp * K_BOLTZMANN_MEV_KELVIN * (1 if meV else 0.001)


def total_resolution_estimate(data: DataType, include_thermal_broadening=False, meV=False):
    """
    Gives the quadrature sum estimate of the resolution of an ARPES
    spectrum that is decorated with appropriate information.

    For synchrotron ARPES, this typically means the scan has the photon energy,
    exit slit information and analyzer slit settings
    :return:
    """

    thermal_broadening = 0
    if include_thermal_broadening:
        thermal_broadening = thermal_broadening_estimate(data, meV=meV)
    return math.sqrt(
        beamline_resolution_estimate(data, meV=meV) ** 2 +
        analyzer_resolution_estimate(data, meV=meV) ** 2 +
        thermal_broadening ** 2
    )