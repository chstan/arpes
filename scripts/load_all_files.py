#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

import arpes.config
from arpes.io import dataset_exists, save_dataset
from arpes.models.spectrum import load_scan
from arpes.utilities import (attach_extra_dataset_columns,
                             clean_datavar_attribute_names,
                             modern_clean_xlsx_dataset,
                             rename_datavar_standard_attrs)
from arpes.utilities.dataset import walk_datasets

sys.path.append(os.getenv('ARPES_ROOT'))
sys.path.append(os.path.join(os.getenv('ARPES_ROOT'), 'arpes'))


DESCRIPTION = """
Command line tool for loading ARPES datasets from spreadsheet. Typical workflow is to call
'load_all_files.py' and then 'load_all_files.py -c' which will also attach the spectrum type 
to the cleaned dataset.
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("-w", "--workspace", help='name of workspace to use (i.e. "RhSn2")')
parser.add_argument("-r", "--reload", help='Reload scans.', action='store_true')
parser.add_argument("-f", "--file",
                    help="specify the dataset that will be used. If it is already clean it will not be cleaned.")
parser.add_argument("-c", "--extra-columns", help="Whether to attach extra columns.", action="store_true")

args = parser.parse_args()
arpes.config.attempt_determine_workspace(args.workspace or os.getenv('WORKSPACE'))

if args.file:
    print("├{}".format(args.file))
    files = [os.path.join(os.getcwd(), args.file)]
else:
    files = walk_datasets()

if args.extra_columns:
    for dataset_path in files:
        attach_extra_dataset_columns(dataset_path)
else:
    for dataset_path in files:
        ds = modern_clean_xlsx_dataset(dataset_path, with_inferred_cols=False)
        print('└┐')
        for file, scan in ds.iterrows():
            print(' ├{}'.format(file))
            scan['file'] = scan.get('path', file)
            if not dataset_exists(scan.get('id')) or args.reload:
                try:
                    import arpes.xarray_extensions

                    data = load_scan(dict(scan))
                    data = rename_datavar_standard_attrs(data.S.spectrum)
                    data = clean_datavar_attribute_names(data)
                    save_dataset(data, force=True)
                except Exception as e:
                    print('Encountered Error {}. Skipping...'.format(e))
