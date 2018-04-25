"""
Store experiment level configuration here, this module also provides functions
for loading configuration in via external files, to allow better modularity
between different projects.
"""

import json
import os.path
import logging

from pathlib import Path

from arpes.exceptions import ConfigurationError
import arpes.constants as consts

# ARPES_ROOT SHOULD BE PROVIDED THROUGH ENVIRONMENT VARIABLES
ARPES_ROOT = os.getenv('ARPES_ROOT')
assert(ARPES_ROOT is not None and "Check to make sure you have the ARPES_ROOT environment "
                                  "variable defined.")

SOURCE_PATH = ARPES_ROOT
FIGURE_PATH = os.path.join(SOURCE_PATH, 'figures')

DATASET_PATH = os.path.join(SOURCE_PATH, 'datasets')
# don't really need this one, but you can set it if you want
EXPERIMENT_PATH = os.path.join(SOURCE_PATH, 'exp')

DATASET_ROOT_PATH = ARPES_ROOT
# these are all set by ``update_configuration``

DATASET_CACHE_PATH = os.path.join(ARPES_ROOT, 'cache')
DATASET_CACHE_RECORD = None
CLEAVE_RECORD = None
CALIBRATION_RECORD = None

PIPELINE_SHELF = None
PIPELINE_JSON_SHELF = None

def update_configuration():
    global DATASET_ROOT_PATH
    global DATASET_CACHE_RECORD
    global CLEAVE_RECORD
    global CALIBRATION_RECORD

    global PIPELINE_SHELF
    global PIPELINE_JSON_SHELF

    DATASET_CACHE_RECORD = os.path.join(DATASET_ROOT_PATH, 'datasets', 'cache.json')
    CLEAVE_RECORD = os.path.join(DATASET_ROOT_PATH, 'datasets', 'cleaves.json')
    CALIBRATION_RECORD = os.path.join(DATASET_ROOT_PATH, 'datasets', 'calibrations.json')

    # TODO use a real database here
    PIPELINE_SHELF = os.path.join(DATASET_ROOT_PATH, 'datasets','pipeline.shelf')
    PIPELINE_JSON_SHELF = os.path.join(DATASET_ROOT_PATH, 'datasets','pipeline.shelf.json')

update_configuration()

CONFIG = {
    'VERSION': '1.0.0',
    'MODE': consts.MODE_ARPES,
    'LATTICE_CONSTANT': consts.LATTICE_CONSTANTS['Bi-2212'],
    'WORK_FUNCTION': 46,
    'LASER_ENERGY': 5.93,
    'WORKSPACE': None, # set me in your notebook before saving anything
}

def workspace_name_is_valid(workspace_name):
    return workspace_name in os.listdir(DATA_PATH)

def workspace_matches(path):
    files = os.listdir(path)
    acceptable_suffixes = {'.xlx', '.xlsx'}

    return 'data' in files and any(Path(f).suffix in acceptable_suffixes for f in files)

def attempt_determine_workspace(value=None, permissive=False):
    # first search upwards from the current directory at most three folders:
    try:
        current_path = os.getcwd()
        for _ in range(3):
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
        current_path = os.path.realpath(os.getcwd())
        option = None
        skip_dirs = {'experiments', 'experiment', 'exp', 'projects', 'project'}

        if os.path.realpath(DATASET_PATH) in current_path:
            path_fragment = current_path.split(os.path.realpath(DATASET_PATH))[1]
            option = [x for x in path_fragment.split('/') if len(x) and x not in skip_dirs][0]
            # we are in a dataset, we can use the folder name in order to configure

        elif os.path.realpath(EXPERIMENT_PATH) in current_path:
            # this doesn't quite work because of symlinks
            path_fragment = current_path.split(os.path.realpath(EXPERIMENT_PATH))[1]
            option = [x for x in path_fragment.split('/') if len(x) and x not in skip_dirs][0]

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


# try to generate cache files if they do not exist
for p in [DATASET_CACHE_RECORD, CLEAVE_RECORD, CALIBRATION_RECORD]:
    fp = Path(p)
    if not fp.exists():
        with open(p, 'w') as f:
            json.dump({}, f)

