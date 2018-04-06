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

if args.file:
    print("├{}".format(args.file))
    files = [os.path.join(os.getcwd, args.file)]
else:
    files = walk_datasets()


for dataset_path in files:
    ds = modern_clean_xlsx_dataset(dataset_path, with_inferred_cols=False, write=True, allow_soft_match=True)

    print('└┐')
    for file, scan in ds.iterrows():
        print(' ├{}'.format(file))
        scan['file'] = scan.get('path', file)
        if not dataset_exists(scan.get('id')) or args.reload:
            try:
                import arpes.xarray_extensions
                data = load_scan(dict(scan))
                data = rename_datavar_standard_attrs(data)
                data = clean_datavar_attribute_names(data)
                save_dataset(data, force=True)
            except Exception as e:
                print('Encountered Error {}. Skipping...'.format(e))

    attach_extra_dataset_columns(dataset_path)