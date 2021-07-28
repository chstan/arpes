import numpy as np
import pytest

from arpes.io import example_data
from arpes.fits.utilities import broadcast_model
from arpes.fits.fit_models import AffineBroadenedFD, QuadraticModel
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.conversion.forward import convert_through_angular_point
import arpes.xarray_extensions


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
            0,
            319.73139835,
            318.12917486,
            258.94653353,
            200.48829069,
            163.12937875,
            346.93136055,
            0,
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
    155, 151, 159, 140, 150, 120, 121, 125, 131, 130, 138, 140, 149, 
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
    assert kx_max == pytest.approx(-0.015037593984962516)
    assert kdata.mean().item() == pytest.approx(613.79029047)
    assert kdata.fillna(0).mean().item() == pytest.approx(415.7048189)


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
        4141.79361851789,
        4352.118805852634,
        4528.183675544601,
        4772.701193743715,
        4967.954937427305,
        5143.416481043858,
        5389.480518039409,
        5564.486620498726,
        5963.2608828950015,
        6495.800810281041,
        6865.562982108332,
        7112.036574537716,
        7796.474181791687,
        8160.106902788172,
        8524.980143784462,
        8520.4603140169,
        8266.738479510586,
        7786.5089626268455,
        7151.2409294143,
        6764.616607333701,
        6381.040104212984,
        6075.501205633937,
        5922.880496514519,
        5625.495181926943,
        3077.8516096096077,
        117.28646806572776,
    ]

    assert kdata.sel(ky=slice(-0.7, 0)).isel(eV=slice(None, -20, 5)).max("ky").values.tolist() == [
        pytest.approx(c) for c in max_values
    ]
