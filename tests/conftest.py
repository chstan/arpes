import os

import pytest

import arpes.config
import arpes.endstations
from tests.utils import load_test_scan


class AttrAccessorDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture(scope='function')
def sandbox_configuration(tmpdir_factory):
    """
    Generates a sandboxed configuration of the ARPES data analysis suite.
    """

    resources_dir = os.path.join(os.getcwd(), 'tests', 'resources')

    def set_workspace(name):
        workspace = {
            'path': os.path.join(resources_dir, 'datasets', name),
            'name': name,
        }
        arpes.config.CONFIG['WORKSPACE'] = workspace

    def load(dataset, id):
        set_workspace(dataset)
        return load_test_scan(dataset, id)

    arpes.config.update_configuration(user_path=resources_dir)
    sandbox = AttrAccessorDict({
        'with_workspace': set_workspace,
        'load': load,
    })
    arpes.config.load_plugins()
    yield sandbox
    arpes.config.CONFIG['WORKSPACE'] = None
    arpes.config.update_configuration(user_path=None)
    arpes.endstations._ENDSTATION_ALIASES = {}
