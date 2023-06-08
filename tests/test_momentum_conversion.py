import numpy as np
import pytest

import arpes.xarray_extensions
from arpes.fits.fit_models import AffineBroadenedFD, QuadraticModel
from arpes.fits.utilities import broadcast_model
from arpes.io import example_data
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.conversion.forward import convert_through_angular_point


def load_energy_corrected():
    fmap = example_data.map.spectrum
    return fmap

    results = broadcast_model(AffineBroadenedFD, cut, "phi", parallelize=False)
    edge = QuadraticModel().guess_fit(results.F.p("fd_center")).eval(x=fmap.phi)
    return fmap.G.shift_by(edge, "eV")


def test_cut_momentum_conversion():
    """Validates that the core APIs are functioning."""
    kdata = convert_to_kspace(example_data.cut.spectrum, kp=np.linspace(-0.12, 0.12, 600))
    selected = kdata.values.ravel()[[0, 200, 800, 1500, 2800, 20000, 40000, 72000]]

    assert np.nan_to_num(selected).tolist() == [
        pytest.approx(c)
        for c in [
            0.0,
            319.7579866213813,
            318.1358154841895,
            258.94674635813885,
            200.47090110761422,
            163.15644326100534,
            346.9402106918347,
            0.0,
        ]
    ]


def test_cut_momentum_conversion_ranges():
    """Validates that the user can select momentum ranges."""

    data = example_data.cut.spectrum
    kdata = convert_to_kspace(data, kp=np.linspace(-0.12, 0.12, 80))

    expected_values = """
    192, 157, 157, 183, 173, 173, 177, 165, 171, 159, 160, 154, 155, 
    153, 146, 139, 139, 138, 127, 125, 121, 117, 118, 113, 125, 145, 
    147, 141, 147, 147, 146, 143, 143, 145, 131, 147, 136, 133, 145, 
    139, 136, 138, 128, 133, 126, 136, 135, 139, 141, 147, 143, 144, 
    155, 151, 159, 140, 150, 120, 121, 125, 127, 130, 138, 140, 149, 
    144, 155, 151, 154, 165, 165, 166, 172, 168, 167, 177, 177, 171, 
    168, 160
    """.replace(
        ",", ""
    ).split()
    expected_values = [int(m) for m in expected_values]
    assert kdata.argmax(dim="eV").values.tolist() == expected_values


def test_fermi_surface_conversion():
    """Validates that the kx-ky conversion code is behaving."""
    data = load_energy_corrected().S.fermi_surface

    kdata = convert_to_kspace(
        data,
        kx=np.linspace(-2.5, 1.5, 400),
        ky=np.linspace(-2, 2, 400),
    )

    kx_max = kdata.idxmax(dim="ky").max().item()
    ky_max = kdata.idxmax(dim="kx").max().item()

    assert ky_max == pytest.approx(0.4373433583959896)
    assert kx_max == pytest.approx(-0.02506265664160412)
    assert kdata.mean().item() == pytest.approx(613.848688084093)
    assert kdata.fillna(0).mean().item() == pytest.approx(415.7673895479573)


@pytest.mark.skip
def test_conversion_with_passthrough_axis():
    """Validates that passthrough is equivalent to individual slice conversion."""
    raise NotImplementedError


@pytest.mark.skip
def test_kz_conversion():
    """Validates the kz conversion code."""
    raise NotImplementedError


@pytest.mark.skip
def test_inner_potential():
    """Validates that the inner potential changes kz offset and kp range."""
    raise NotImplementedError


@pytest.mark.skip
def test_convert_angular_pair():
    """Validates that we correctly convert through high symmetry points and angle."""
    raise NotImplementedError


def test_convert_angular_point_and_angle():
    """Validates that we correctly convert through high symmetry points."""

    test_point = {
        "phi": -0.13,
        "theta": -0.1,
        "eV": 0.0,
    }
    data = load_energy_corrected()

    kdata = convert_through_angular_point(
        data,
        test_point,
        {"ky": np.linspace(-1, 1, 400)},
        {"kx": np.linspace(-0.02, 0.02, 10)},
    )

    max_values = [
        4141.827366031282,
        4352.104413953421,
        4528.141587081243,
        4772.790364388664,
        4967.805454675143,
        5143.319351060313,
        5389.489299730738,
        5564.495169531284,
        5963.146620422676,
        6495.75206989827,
        6865.545155007645,
        7112.055898285905,
        7796.474144588328,
        8160.193714723893,
        8525.136971985057,
        8520.552639235233,
        8266.861947781663,
        7786.596026041245,
        7151.341160693082,
        6764.770214858431,
        6381.080528876631,
        6075.551683306253,
        5922.880032224642,
        5625.561944388922,
        3077.8859544793277,
        117.28906072530499,
    ]

    assert kdata.sel(ky=slice(-0.7, 0)).isel(eV=slice(None, -20, 5)).max("ky").values.tolist() == [
        pytest.approx(c) for c in max_values
    ]
