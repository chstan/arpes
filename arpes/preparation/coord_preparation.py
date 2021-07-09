"""Utilities related to treating coordinates during data prep."""
import collections
import functools

import numpy as np


__all__ = ["disambiguate_coordinates"]


def disambiguate_coordinates(datasets, possibly_clashing_coordinates):
    """Finds and unifies duplicated coordinates or ambiguous coordinates.

    This is useful if two regions claim to have an energy axis, but one is a core level
    and so refers to a different energy range.
    """
    coords_set = collections.defaultdict(list)
    for d in datasets:
        for c in possibly_clashing_coordinates:
            if c in d.coords:
                coords_set[c].append(d.coords[c])

    conflicted = []
    for c in possibly_clashing_coordinates:
        different_coords = coords_set[c]
        if not different_coords:
            continue

        if not functools.reduce(
            lambda x, y: (np.array_equal(x[1], y) and x[0], y),
            different_coords,
            (True, different_coords[0]),
        )[0]:
            conflicted.append(c)

    after_deconflict = []
    for d in datasets:
        spectrum_name = list(d.data_vars.keys())[0]
        to_rename = {name: name + "-" + spectrum_name for name in d.dims if name in conflicted}
        after_deconflict.append(d.rename(to_rename))

    return after_deconflict
