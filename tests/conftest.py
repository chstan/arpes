import os

import pytest

import arpes.config
import arpes.endstations
from tests.utils import cache_loader


class AttrAccessorDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


SCAN_FIXTURE_LOCATIONS = {
    "basic/main_chamber_cut_0.fits": "ALG-MC",
    "basic/main_chamber_map_1.fits": "ALG-MC",
    "basic/main_chamber_PHONY_2.fits": "ALG-MC",
    "basic/main_chamber_PHONY_3.fits": "ALG-MC",
    "basic/SToF_PHONY_4.fits": "ALG-SToF",
    "basic/SToF_PHONY_5.fits": "ALG-SToF",
    "basic/SToF_PHONY_6.fits": "ALG-SToF",
    "basic/SToF_PHONY_7.fits": "ALG-SToF",
    "basic/MERLIN_8.pxt": "BL4",
    "basic/MERLIN_9.pxt": "BL4",
    "basic/MERLIN_10_S001.pxt": "BL4",
    "basic/MERLIN_11_S001.pxt": "BL4",
    "basic/MAESTRO_12.fits": "BL7",
    "basic/MAESTRO_13.fits": "BL7",
    "basic/MAESTRO_PHONY_14.fits": "BL7",
    "basic/MAESTRO_PHONY_15.fits": "BL7",
    "basic/MAESTRO_16.fits": "BL7",
    "basic/MAESTRO_nARPES_focus_17.fits": "BL7-nano",
}


@pytest.fixture(scope="function")
def sandbox_configuration(tmpdir_factory):
    """
    Generates a sandboxed configuration of the ARPES data analysis suite.
    """

    resources_dir = os.path.join(os.getcwd(), "tests", "resources")

    def set_workspace(name):
        workspace = {
            "path": os.path.join(resources_dir, "datasets", name),
            "name": name,
        }
        arpes.config.CONFIG["WORKSPACE"] = workspace

    def load(path):
        assert path in SCAN_FIXTURE_LOCATIONS
        pieces = path.split("/")
        set_workspace(pieces[0])
        return cache_loader.load_test_scan(
            os.path.join(*pieces), location=SCAN_FIXTURE_LOCATIONS[path]
        )

    arpes.config.update_configuration(user_path=resources_dir)
    sandbox = AttrAccessorDict(
        {
            "with_workspace": set_workspace,
            "load": load,
        }
    )
    arpes.config.load_plugins()
    yield sandbox
    arpes.config.CONFIG["WORKSPACE"] = None
    arpes.config.update_configuration(user_path=None)
    arpes.endstations._ENDSTATION_ALIASES = {}
