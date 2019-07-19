import pytest
import warnings

import arpes.config


def load_test_scan(dataset_name, file_number):
    from arpes.utilities.dataset import default_dataset
    from arpes.io import direct_load

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        df = default_dataset(match=dataset_name, write=False)
        return direct_load(file_number, df=df)
