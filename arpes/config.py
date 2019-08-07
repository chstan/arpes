"""
Store experiment level configuration here, this module also provides functions
for loading configuration in via external files, to allow better modularity
between different projects.
"""

# pylint: disable=global-statement


import json
import logging
import os.path
import pint

from pathlib import Path
from typing import Any, Optional

from arpes.exceptions import ConfigurationError

ureg = pint.UnitRegistry()

# ARPES_ROOT SHOULD BE PROVIDED THROUGH ENVIRONMENT VARIABLES, or via `setup`
DATA_PATH = None
SOURCE_ROOT = str(Path(__file__).parent)

SETTINGS = {
    'interactive': {
        'main_width': 350,
        'marginal_width': 150,
        'palette': 'magma',
    },
    'xarray_repr_mod': False,
}

# these are all set by ``update_configuration``
FIGURE_PATH = None
DATASET_PATH = None

DATASET_CACHE_PATH = None
DATASET_CACHE_RECORD = None # .json file that holds normalized files

# .json file that records which files are linked to the same physical sample, currently unused
CLEAVE_RECORD = None

PIPELINE_SHELF = None
PIPELINE_JSON_SHELF = None


def generate_cache_files() -> None:
    global DATASET_CACHE_RECORD
    global CLEAVE_RECORD

    for record_file in [DATASET_CACHE_RECORD, CLEAVE_RECORD]:
        if not Path(record_file).exists():
            with open(record_file, 'w') as f:
                json.dump({}, f)


def update_configuration(user_path: Optional[str] = None) -> None:
    global FIGURE_PATH
    global DATASET_PATH
    global DATASET_CACHE_PATH
    global DATASET_CACHE_RECORD
    global CLEAVE_RECORD
    global PIPELINE_SHELF
    global PIPELINE_JSON_SHELF

    try:
        FIGURE_PATH = os.path.join(user_path, 'figures')
        DATASET_PATH = os.path.join(user_path, 'datasets')

        DATASET_CACHE_PATH = os.path.join(user_path, 'cache')

        DATASET_CACHE_RECORD = os.path.join(user_path, 'datasets', 'cache.json')
        CLEAVE_RECORD = os.path.join(user_path, 'datasets', 'cleaves.json')

        PIPELINE_SHELF = os.path.join(user_path, 'datasets', 'pipeline.shelf')
        PIPELINE_JSON_SHELF = os.path.join(user_path, 'datasets', 'pipeline.shelf.json')

        generate_cache_files()
    except TypeError:
        pass


CONFIG = {
    'WORKSPACE': {},
    'CURRENT_CONTEXT': None,
}


class WorkspaceManager:
    def __init__(self, workspace: Optional[Any] = None) -> None:
        self._cached_workspace = None
        self._workspace = workspace

    def __enter__(self) -> None:
        global CONFIG
        self._cached_workspace = CONFIG['WORKSPACE']

        if not self._workspace:
            return

        if not CONFIG['WORKSPACE']:
            attempt_determine_workspace()

        workspace_path = Path(CONFIG['WORKSPACE']['path']).parent / self._workspace

        if workspace_path.exists():
            CONFIG['WORKSPACE'] = dict(CONFIG['WORKSPACE'])
            CONFIG['WORKSPACE']['name'] = self._workspace
            CONFIG['WORKSPACE']['path'] = str(workspace_path)
        else:
            raise ValueError('Could not find workspace: {}'.format(self._workspace))

    def __exit__(self, *args: Any) -> None:
        global CONFIG
        CONFIG['WORKSPACE'] = self._cached_workspace


def workspace_name_is_valid(workspace_name):
    return workspace_name in os.listdir(DATA_PATH)


def workspace_matches(path):
    files = os.listdir(path)
    acceptable_suffixes = {'.xlx', '.xlsx', '.numbers'}

    return 'data' in files and any(Path(f).suffix in acceptable_suffixes for f in files)


def attempt_determine_workspace(value=None, permissive=False, lazy=False, current_path=None):
    # first search upwards from the current directory at most three folders:
    if lazy and not CONFIG['WORKSPACE']:
        return

    try:
        current_path = os.getcwd()
        for _ in range(3):
            print('Checking: {}'.format(current_path))
            if workspace_matches(current_path):
                CONFIG['WORKSPACE'] = {
                    'path': current_path,
                    'name': Path(current_path).name
                }
                return

            current_path = Path(current_path).parent
    except Exception: # pylint: disable=broad-except
        pass

    if not CONFIG['WORKSPACE']:
        if current_path is None:
            current_path = os.path.realpath(os.getcwd())

        option = None
        skip_dirs = {'experiments', 'experiment', 'exp', 'projects', 'project'}

        if os.path.realpath(DATASET_PATH) in str(current_path):
            path_fragment = current_path.split(os.path.realpath(DATASET_PATH))[1]
            option = [x for x in path_fragment.split('/') if len(x) and x not in skip_dirs][0]
            # we are in a dataset, we can use the folder name in order to configure

        if value is not None:
            option = value

        if workspace_name_is_valid(option):
            logging.warning('Automatically inferring that the workspace is "{}"'.format(option))
            CONFIG['WORKSPACE'] = option

    if not CONFIG['WORKSPACE'] and not permissive:
        raise ConfigurationError('You must provide a workspace.')


def load_json_configuration(filename):
    """
    Flat updates the configuration. Beware that this doesn't update nested data.
    I will adjust if it turns out that there is a use case for nested configuration
    """
    with open(filename) as config_file:
        CONFIG.update(json.load(config_file))


try:
    from local_config import * # pylint: disable=wildcard-import
except ImportError:
    logging.warning("Could not find local configuration file. If you don't "
                    "have one, you can safely ignore this message.")


def override_settings(new_settings):
    from arpes.utilities.collections import deep_update

    global SETTINGS
    deep_update(SETTINGS, new_settings)


# try to generate cache files if they do not exist
try:
    generate_cache_files()
except TypeError:
    pass


# load plugins
def load_plugins() -> None:
    import arpes.endstations.plugin as plugin
    from arpes.endstations import add_endstation
    import importlib

    skip_modules = {'__pycache__', '__init__'}
    plugins_dir = str(Path(plugin.__file__).parent)
    modules = os.listdir(plugins_dir)
    modules = [m if os.path.isdir(os.path.join(plugins_dir, m))
               else os.path.splitext(m)[0] for m in modules if m not in skip_modules]

    for module in modules:
        try:
            loaded_module = importlib.import_module('arpes.endstations.plugin.{}'.format(module))
            for item in loaded_module.__all__:
                add_endstation(getattr(loaded_module, item))
        except (AttributeError, ImportError):
            pass


def use_tex(rc_text_should_use=False):
    import matplotlib
    matplotlib.rcParams['text.usetex'] = rc_text_should_use


update_configuration()
load_plugins()
