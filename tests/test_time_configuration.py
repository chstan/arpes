import os.path

import arpes.config


def test_patched_config(sandbox_configuration):
    sandbox_configuration.with_workspace("basic")
    assert arpes.config.CONFIG["WORKSPACE"]["name"] == "basic"
    assert arpes.config.CONFIG["WORKSPACE"]["path"].split(os.sep)[-2:] == ["datasets", "basic"]


def test_patched_config_no_workspace(sandbox_configuration):
    assert arpes.config.CONFIG["WORKSPACE"] is None
