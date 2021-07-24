"""Provides standard corrections for datasets.

Largely, this covers:
1. Fermi edge corrections
2. Background estimation and subtraction

It also contains utilities related to identifying a piece of data
earlier in a dataset which can be used to furnish equivalent references.

"""
from collections import OrderedDict

from arpes.typing import DataType
from arpes.utilities import normalize_to_dataset, deep_equals
from .fermi_edge_corrections import *


__all__ = (
    "reference_key",
    "correction_from_reference_set",
)


class HashableDict(OrderedDict):
    """Implements hashing for ordered dictionaries.

    The dictionary must be ordered for the hash to be stable.
    """

    def __hash__(self):
        return hash(frozenset(self.items()))


def reference_key(data: DataType):
    """Calculates a key/hash for data determining reference/correction equality."""
    data = normalize_to_dataset(data)

    return HashableDict(data.S.reference_settings)


def correction_from_reference_set(data: DataType, reference_set):
    """Determines which correction to use from a set of references."""
    data = normalize_to_dataset(data)

    correction = None
    for k, corr in reference_set.items():
        if deep_equals(dict(reference_key(data)), dict(k)):
            correction = corr
            break

    return correction
