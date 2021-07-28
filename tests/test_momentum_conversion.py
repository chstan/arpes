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
    cut = fmap.sum("theta").sel(eV=slice(-0.2, 0.1), phi=slice(-0.25, 0.3))

    results = broadcast_model(AffineBroadenedFD, cut, "phi")
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
    assert kdata.mean().item() == pytest.approx(820.5092717)
    assert kdata.fillna(0).mean().item() == pytest.approx(555.710417022)


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
        0.0,
        4332.821244449737,
        4532.076642909317,
        4753.245322434983,
        4933.894741353865,
        5145.826374585494,
        5389.024768971414,
        5560.16820212156,
        5934.7580649508045,
        6477.344494538664,
        6848.5894673342045,
        7112.208395216838,
        7742.842693967063,
        8141.299758744744,
        8489.459016788936,
        8529.650628690946,
        8277.185652641712,
        7802.5196797546305,
        7171.889758677037,
        6792.642286906545,
        6378.868806982989,
        6087.525338632965,
        5916.242988353441,
        5644.807978079201,
        3251.882137466492,
        141.21390323116498,
    ]

    assert kdata.sel(ky=slice(-0.7, 0)).isel(eV=slice(None, -20, 5)).max("ky").values.tolist() == [
        pytest.approx(c) for c in max_values
    ]
