#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import os
import sys

import uuid
from os import walk

sys.path.append(os.getenv('ARPES_ROOT'))

import arpes.config
from arpes.models.spectrum import load_scan
from arpes.utilities import clean_xlsx_dataset, attach_extra_dataset_columns, \
    rename_datavar_standard_attrs, clean_datavar_attribute_names
from arpes.io import save_dataset
from arpes.exceptions import ConfigurationError

def attach_uuid(scan):
    if 'id' not in scan:
        scan = copy.copy(scan)
        scan['id'] = str(uuid.uuid1())

    return scan


_SEARCH_FOLDERS = {'hdf5', 'fits', }

DESCRIPTION = """
Command line tool for loading ARPES datasets from spreadsheet. Typical workflow is to call
'load_all_files.py -w {WORKSPACE} -l' and then 'load_all_files.py -w {WORKSPACE} -uc -c'
which will also attach the spectrum type to the cleaned dataset. You will need to do the last step
before you can expect 'default_dataset()' to work.
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("-c", "--attach-columns", help="load datasets to attach extra columns to cleaned sheet",
                    action="store_true")
parser.add_argument("-w", "--workspace", help='name of workspace to use (i.e. "RhSn2")')
parser.add_argument("-l", "--load", help="flag to load data from original source and save as NetCDF",
                    action="store_true")
parser.add_argument("-f", "--file", help="specify the dataset that will be used. If it is already clean it will not be cleaned.")
parser.add_argument("-uc", "--use-clean", help="skip dataset cleaning process", action="store_true")

args = parser.parse_args()

def is_excel(f):
    return '.xlsx' in f or '.xlx' in f

def is_cleaned(f, path=None):
    return 'cleaned' in x or (isinstance(path, str) and 'cleaned' in path)

if args.file:
    print("Preparing single file {}".format(args.file))
    assert(is_excel(args.file))

    current_path = os.getcwd()
    ds = clean_xlsx_dataset(os.path.join(current_path, args.file),
                            with_inferred_cols=False, reload=False)

    for file, scan in ds.iterrows():
        print("├{}".format(file))
        scan['file'] = scan.get('path', file)
        try:
            data = load_scan(dict(scan))
            data = rename_datavar_standard_attrs(data)
            data = clean_datavar_attribute_names(data)
            save_dataset(data, force=True)
        except Exception as e:
            print('Encountered Error {}. Skipping...'.format(e))

else:
    if arpes.config.CONFIG['WORKSPACE'] is None:
        arpes.config.CONFIG['WORKSPACE'] = args.workspace or os.getenv('WORKSPACE')

    if arpes.config.CONFIG['WORKSPACE'] is None:
        raise ConfigurationError('You must provide a workspace.')

    if args.load:
        for path, _, files in walk(os.getcwd()):
            excel_files = [f for f in files if is_excel(f)]

            for x in excel_files:
                print(x)
                if args.use_clean != (is_cleaned(x, path)):
                    print('SKIPPING\n')
                    continue

                ds = clean_xlsx_dataset(os.path.join(path, x), with_inferred_cols=False, reload=not args.use_clean)

                for file, scan in ds.iterrows():
                    print("├{}".format(file))
                    scan['file'] = scan.get('path', file)
                    try:
                        import arpes.xarray_extensions
                        data = load_scan(dict(scan))
                        data = rename_datavar_standard_attrs(data.S.spectrum)
                        data = clean_datavar_attribute_names(data)
                        save_dataset(data, force=True)
                    except Exception as e:
                        print('Encountered Error {}. Skipping...'.format(e))

                print()

    if args.attach_columns:
        assert(args.use_clean)
        for path, _, files in walk(os.getcwd()):
            # JSON files are deprecated
            excel_files = [f for f in files if is_excel(f)]

            for x in excel_files:
                if args.use_clean != (is_cleaned(x, path)):
                    continue

                attach_extra_dataset_columns(os.path.join(path, x))
