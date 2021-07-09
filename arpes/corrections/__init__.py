from collections import OrderedDict

from arpes.corrections.fermi_edge_corrections import apply_quadratic_fermi_edge_correction
from arpes.typing import DataType
from arpes.utilities import normalize_to_dataset, normalize_to_spectrum, deep_equals
from .fermi_edge_corrections import *


__all__ = (
    "reference_key",
    "correction_from_reference_set",
)


class HashableDict(OrderedDict):
    """
    Implements hashing for ordered dictionaries. The dictionary
    must be ordered for the hash to be stable.
    """

    def __hash__(self):
        return hash(frozenset(self.items()))


def reference_key(data: DataType):
    data = normalize_to_dataset(data)

    return HashableDict(data.S.reference_settings)


def correction_from_reference_set(data: DataType, reference_set):
    data = normalize_to_dataset(data)

    correction = None
    for k, corr in reference_set.items():
        if deep_equals(dict(reference_key(data)), dict(k)):
            correction = corr
            break

    return correction
