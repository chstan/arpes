"""
Store experiment level configuration here, this module also provides functions
for loading configuration in via external files, to allow better modularity
between different projects.
"""

import json
import os.path
import logging
import pint
import collections

from pathlib import Path

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

def generate_cache_files():
    global DATASET_CACHE_RECORD
    global CLEAVE_RECORD

    for p in [DATASET_CACHE_RECORD, CLEAVE_RECORD]:
        fp = Path(p)
        if not fp.exists():
            with open(p, 'w') as f:
                json.dump({}, f)

def update_configuration(user_path=None):
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

        # TODO use a real database here
        PIPELINE_SHELF = os.path.join(user_path, 'datasets','pipeline.shelf')
        PIPELINE_JSON_SHELF = os.path.join(user_path, 'datasets','pipeline.shelf.json')

        generate_cache_files()
    except TypeError:
        pass



update_configuration()

CONFIG = {
    'WORKSPACE': None,
}


class WorkspaceManager(object):
    def __init__(self, workspace=None):
        self._cached_workspace = None
        self._workspace = workspace

    def __enter__(self):
        self._cached_workspace = CONFIG['WORKSPACE']

        if self._workspace is None:
            return

        if CONFIG['WORKSPACE'] is None:
            attempt_determine_workspace()

        p = Path(CONFIG['WORKSPACE']['path']).parent / self._workspace

        if p.exists():
            CONFIG['WORKSPACE'] = dict(CONFIG['WORKSPACE'])
            CONFIG['WORKSPACE']['name'] = self._workspace
            CONFIG['WORKSPACE']['path'] = str(p)
        else:
            raise ValueError('Could not find workspace: {}'.format(self._workspace))

    def __exit__(self, *args):
        CONFIG['WORKSPACE'] = self._cached_workspace


def workspace_name_is_valid(workspace_name):
    return workspace_name in os.listdir(DATA_PATH)


def workspace_matches(path):
    files = os.listdir(path)
    acceptable_suffixes = {'.xlx', '.xlsx', '.numbers'}

    return 'data' in files and any(Path(f).suffix in acceptable_suffixes for f in files)


def attempt_determine_workspace(value=None, permissive=False, lazy=False, current_path=None):
    # first search upwards from the current directory at most three folders:
    if lazy and CONFIG['WORKSPACE'] is not None:
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
    except Exception:
        pass

    if CONFIG['WORKSPACE'] is None:
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

    if CONFIG['WORKSPACE'] is None and not permissive:
        raise ConfigurationError('You must provide a workspace.')


def load_json_configuration(filename):
    """
    Flat updates the configuration. Beware that this doesn't update nested data.
    I will adjust if it turns out that there is a use case for nested configuration
    """
    with open(filename) as config_file:
        CONFIG.update(json.load(config_file))


try:
    from local_config import *
except:
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
def load_plugins():
    import arpes.endstations.plugin as plugin
    from arpes.endstations import add_endstation
    import importlib
    from pathlib import Path

    skip_modules = {'__pycache__', '__init__'}
    plugins_dir = str(Path(plugin.__file__).parent)
    modules = os.listdir(plugins_dir)
    modules = [m if os.path.isdir(os.path.join(plugins_dir, m))
               else os.path.splitext(m)[0] for m in modules if m not in skip_modules]

    endstation_classes = {}
    for module in modules:
        try:
            loaded_module = importlib.import_module('arpes.endstations.plugin.{}'.format(module))
            for item in loaded_module.__all__:
                add_endstation(getattr(loaded_module, item))
            #plugin_cls = loaded_module
            #experiment_classes[module] = loaded_module.Experiment
        except (AttributeError, ImportError) as e:
            pass


def use_tex(rc_text_should_use=False):
    import matplotlib
    matplotlib.rcParams['text.usetex'] = rc_text_should_use
