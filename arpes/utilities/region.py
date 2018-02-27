from enum import Enum
from typing import Union

__all__ = ['REGIONS', 'normalize_region', 'DesignatedRegions']


class DesignatedRegions(Enum):
    # angular windows
    NARROW_ANGLE = 0 # trim to a narrow central region in the spectrometer
    WIDE_ANGLE = 1 # trim to just inside edges of spectrometer data
    TRIM_EMPTY = 2 # trim to edges of spectrometer data

    # energy windows
    BELOW_EF = 10  # everything below e_F
    ABOVE_EF = 11  # everything above e_F
    EF_NARROW = 12  # narrow cut around e_F
    MESO_EF = 13 # comfortably below e_F, pun on mesosphere

    # effective energy windows, determined by Canny edge detection
    BELOW_EFFECTIVE_EF = 20  # everything below e_F
    ABOVE_EFFECTIVE_EF = 21  # everything above e_F
    EFFECTIVE_EF_NARROW = 22  # narrow cut around e_F
    MESO_EFFECTIVE_EF = 23 # comfortably below effective e_F, pun on mesosphere


REGIONS = {
    'copper_prior': {
        'eV': DesignatedRegions.MESO_EFFECTIVE_EF,
    },
    # angular can refer to either 'pixels' or 'phi'
    'wide_angular': {
        # angular can refer to either 'pixels' or 'phi'
        'angular': DesignatedRegions.WIDE_ANGLE,
    },
    'narrow_angular': {
        'angular': DesignatedRegions.NARROW_ANGLE,
    },
}


def normalize_region(region: Union[str, dict]):
    if isinstance(region, str):
        return REGIONS[region]

    if isinstance(region, dict):
        return region

    raise TypeError('Region should be either a string (i.e. an ID/alias) '
                    'or an explicit dictionary.')