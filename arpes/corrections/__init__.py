from collections import OrderedDict
import pandas as pd

from arpes.analysis import apply_mask
from arpes.corrections.background import remove_incoherent_background
from arpes.corrections.fermi_edge_corrections import apply_quadratic_fermi_edge_correction
from arpes.typing import DataType
from arpes.utilities import normalize_to_dataset, normalize_to_spectrum, deep_equals
from .fermi_edge_corrections import *
from .cycle import *


from arpes.io import simple_load

__all__ = ('build_reference_set', 'reference_key', 'apply_from_reference_set',
           'correction_from_reference_set',)


class HashableDict(OrderedDict):
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


def apply_from_reference_set(data: DataType, reference_set, **kwargs):
    correction = correction_from_reference_set(data, reference_set)

    data = normalize_to_spectrum(data)
    if correction is None:
        return data

    return apply_quadratic_fermi_edge_correction(data, correction, **kwargs)


def build_reference_set(df: pd.DataFrame, mask=None):
    references = {}

    for index, col in df.iterrows():
        print(index)
        data = simple_load(int(index), df)
        settings = data.S.reference_settings

        data = data.spectrum.S.sum_other(['phi', 'eV'])

        if mask is not None:
            data = apply_mask(data, mask, radius=-15)

        data = remove_incoherent_background(data)
        correction = build_quadratic_fermi_edge_correction(arr=data, fit_limit=0.0001)

        references[HashableDict(settings)] = correction

    return references
