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
ARPES_ROOT = os.path.getenv('ARPES_ROOT')
assert(ARPES_ROOT is not None and "Check to make sure you have the ARPES_ROOT environment "
                                  "variable defined.")

SOURCE_PATH = ARPES_ROOT
FIGURE_PATH = os.path.join(SOURCE_PATH, 'figures/')

DATASET_CACHE_RECORD = os.path.join(SOURCE_PATH, 'datasets/cache.json')
CLEAVE_RECORD = os.path.join(SOURCE_PATH, 'datasets/cleaves.json')
CALIBRATION_RECORD = os.path.join(SOURCE_PATH, 'datasets/calibrations.json')

# TODO use a real database here
PIPELINE_SHELF = os.path.join(SOURCE_PATH, 'datasets/pipeline.shelf')
PIPELINE_JSON_SHELF = os.path.join(SOURCE_PATH, 'datasets/pipeline.shelf.json')

CONFIG = {
    'VERSION': '1.0.0',
    'MODE': consts.MODE_ARPES,
    'LATTICE_CONSTANT': consts.LATTICE_CONSTANTS['Bi-2212'],
    'WORK_FUNCTION': 46,
    'LASER_ENERGY': 5.93,
    'WORKSPACE': None, # set me in your notebook before saving anything
}

def load_json_configuration(filename):
    """
    Flat updates the configuration. Beware that this doesn't update nested data.
    I will adjust if it turns out that there is a use case for nested configuration
    """
    with open(filename) as config_file:
        CONFIG.update(json.load(config_file))


try:
    import local_config
except:
    warnings.warn("Could not find local configuration file. If you don't "
                  "have one, you can safely ignore this message.")