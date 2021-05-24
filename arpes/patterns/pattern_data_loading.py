from .pattern_imports import *


def simple_implicit_dataset():
    # loads the file f00034 or similar, looks at end of name for match
    f34 = simple_load(34)


def dataset_loading():
    ds = default_dataset()


def dataset_loading_disambiguated():
    # loads 120818.cleaned.xlsx
    ds = default_dataset(match="0818")


def simple_explicit_dataset():
    ds = default_dataset(match="0818")

    # loads f00034 as referenced in 120818.cleaned.xlsx, as opposed to
    # any other spreadsheet
    f34 = simple_load(34, ds)
