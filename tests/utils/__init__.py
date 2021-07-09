import pytest
import warnings

import arpes.config
from arpes.io import load_example_data

__all__ = ["load_test_scan"]


def load_test_scan(example_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return load_example_data(example_name)
