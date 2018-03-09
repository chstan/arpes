"""
Store experiment level configuration here, this module also provides functions
for loading configuration in via external files, to allow better modularity
between different projects.
"""

import json
import os.path
import warnings

import arpes.constants as consts

# ARPES_ROOT SHOULD BE PROVIDED THROUGH ENVIRONMENT VARIABLES
ARPES_ROOT = os.getenv('ARPES_ROOT')
assert(ARPES_ROOT is not None and "Check to make sure you have the ARPES_ROOT environment "
                                  "variable defined.")

SOURCE_PATH = ARPES_ROOT
FIGURE_PATH = os.path.join(SOURCE_PATH, 'figures/')

DATASET_PATH = os.path.join(SOURCE_PATH, 'datasets')
# don't really need this one, but you can set it if you want
EXPERIMENT_PATH = os.path.join(SOURCE_PATH, 'exp')

DATASET_ROOT_PATH = ARPES_ROOT
# these are all set by ``update_configuration``
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

    DATASET_CACHE_RECORD = os.path.join(DATASET_ROOT_PATH, 'datasets/cache.json')
    CLEAVE_RECORD = os.path.join(DATASET_ROOT_PATH, 'datasets/cleaves.json')
    CALIBRATION_RECORD = os.path.join(DATASET_ROOT_PATH, 'datasets/calibrations.json')

    # TODO use a real database here
    PIPELINE_SHELF = os.path.join(DATASET_ROOT_PATH, 'datasets/pipeline.shelf')
    PIPELINE_JSON_SHELF = os.path.join(DATASET_ROOT_PATH, 'datasets/pipeline.shelf.json')

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

def attempt_determine_workspace():
    if CONFIG['WORKSPACE'] is None:
        current_path = os.path.realpath(os.getcwd())
        print(current_path)

        option = None
        skip_dirs = {'experiments', 'experiment', 'exp', 'projects', 'project'}

        if os.path.realpath(DATASET_ROOT_PATH) in current_path:
            path_fragment = current_path.split(os.path.realpath(DATASET_ROOT_PATH))[1]
            option = [x for x in path_fragment.split('/') if len(x) and x not in skip_dirs][0]
            # we are in a dataset, we can use the folder name in order to configure

        elif os.path.realpath(EXPERIMENT_PATH) in current_path:
            # this doesn't quite work because of symlinks
            path_fragment = current_path.split(os.path.realpath(EXPERIMENT_PATH))[1]
            option = [x for x in path_fragment.split('/') if len(x) and x not in skip_dirs][0]

        if workspace_name_is_valid(option):
            warnings.warn('Automatically inferring that the workspace is {}'.format(option))
            CONFIG['WORKSPACE'] = option

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
    warnings.warn("Could not find local configuration file. If you don't "
                  "have one, you can safely ignore this message.")