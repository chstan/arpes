#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

sys.path.append(os.getenv('ARPES_ROOT'))
sys.path.append(os.path.join(os.getenv('ARPES_ROOT'), 'arpes'))

try:
    import arpes.config
    from arpes.models.spectrum import load_scan
    from arpes.utilities import modern_clean_xlsx_dataset, \
        attach_extra_dataset_columns, rename_datavar_standard_attrs, \
        clean_datavar_attribute_names
    from arpes.utilities.dataset import walk_datasets
    from arpes.io import save_dataset, dataset_exists
    from arpes.utilities.autoprep import prepare_raw_files
except ImportError as e:
    print('Did you forget to start your virtual environment?\nImport Error: {}'.format(e))
    raise(e)

DESCRIPTION = """
Command line tool for loading ARPES datasets from spreadsheet. Typical workflow is to call
'autoprep.py' with no arguments. You can see what arguments can be used to customize behavior
with --help. 
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("-w", "--workspace", help='name of workspace to use (i.e. "RhSn2")')
parser.add_argument("-r", "--reload", help='Reload scans.', action='store_true')
parser.add_argument("-f", "--file", help="specify the dataset that will be used. If it is already clean it will not be cleaned.")

args = parser.parse_args()

arpes.config.attempt_determine_workspace(args.workspace or os.getenv('WORKSPACE'))
prepare_raw_files(workspace=None, reload=args.reload, file=args.file)
