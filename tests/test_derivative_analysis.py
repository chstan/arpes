import pytest
from arpes.analysis.derivative import dn_along_axis
from arpes.analysis.filters import gaussian_filter_arr


def test_dataarray_derivatives(sandbox_configuration):
    """
    Nick ran into an issue where he could not call dn_along_axis with a smooth function that
    expected a DataArray, but this is supposed to be supported. Issue was translation between np.ndarray
    and xr.DataArray internal to dn_along_axis.

    :param sandbox_configuration:
    :return:
    """
    def wrapped_filter(arr):
        return gaussian_filter_arr(arr, {'eV': 0.05, 'pixel': 10})

    data = sandbox_configuration.load('basic', 0).spectrum

    d2_data = dn_along_axis(data, 'eV', wrapped_filter, order=2)

    # some random sample
    assert [pytest.approx(v, 1e-3) for v in (d2_data.values[50:55, 60:62].ravel())] == [
        28007.35, 27926.49, 28041.94, 27925.58, 28014.83,
        27862.66, 27925.76, 27737.75, 27775.20, 27551.59,
    ]
