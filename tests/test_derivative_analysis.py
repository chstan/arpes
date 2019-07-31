import numpy as np

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
        return gaussian_filter_arr(arr, {'eV': 0.05, 'phi': np.pi / 180})

    data = sandbox_configuration.load('basic', 0).spectrum

    d2_data = dn_along_axis(data, 'eV', wrapped_filter, order=2)

    # some random sample
    assert [pytest.approx(v, 1e-3) for v in (d2_data.values[50:55, 60:62].ravel())] == [
        29298.377691428424, 29230.839777046887,
        29354.147718455544, 29245.10370224837,
        29331.912863686706, 29180.932258864683,
        29231.45715140855, 29038.482805082265,
        29053.593654735683, 28818.9411615606841
    ]
