import pytest
import numpy as np
from arpes.io import example_data
from arpes.analysis.general import rebin
from arpes.fits.utilities import broadcast_model
from arpes.fits.fit_models import AffineBroadenedFD


def test_broadcast_fitting():
    cut = example_data.cut.spectrum
    near_ef = cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
    near_ef = rebin(near_ef, phi=5)

    fit_results = broadcast_model([AffineBroadenedFD], near_ef, "phi")

    assert np.abs(fit_results.F.p("a_fd_center").values.mean() + 0.00287) < 1e-4
