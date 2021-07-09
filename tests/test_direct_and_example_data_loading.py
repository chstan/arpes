from pathlib import Path

import xarray as xr
from arpes.io import load_example_data, load_data


def test_load_data(sandbox_configuration):
    test_data_location = (
        Path(__file__).parent
        / "resources"
        / "datasets"
        / "basic"
        / "data"
        / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)


def test_load_example_data(sandbox_configuration):
    data = load_example_data()

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)
