from pathlib import Path

import xarray as xr
import numpy as np
from arpes.io import load_example_data, load_data
from arpes.endstations.plugin.ALG_main import ALGMainChamber


def test_load_data(sandbox_configuration):
    test_data_location = (
        Path(__file__).parent / "resources" / "datasets" / "basic" / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)


def test_load_data_with_plugin_specified(sandbox_configuration):
    test_data_location = (
        Path(__file__).parent / "resources" / "datasets" / "basic" / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")
    directly_specified_data = load_data(file=test_data_location, location=ALGMainChamber)

    assert isinstance(directly_specified_data, xr.Dataset)
    assert directly_specified_data.spectrum.shape == (240, 240)
    assert np.all(data.spectrum.values == directly_specified_data.spectrum.values)


def test_load_example_data(sandbox_configuration):
    data = load_example_data("cut")

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)
