#!/usr/bin/env python

import argparse
import os
import sys

from os import walk

sys.path.append(os.getenv('ARPES_ROOT'))

from arpes.utilities import clean_xlsx_dataset, cleaned_dataset_exists
from arpes.exceptions import ConfigurationError
import arpes.config


DESCRIPTION = """
Command line tool to transform user readable dataset spreadsheets into ones that are automatically parseable
by the analysis code. Typical workflow is to call 'clean_dataset.py'. If you need to specify the header offset,
you can do with with UNFINISHED.
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '-o', '--offset', help="Line offset for header. This should equal the line on which the header appears "
                           "in the file", type=int, default=1)
parser.add_argument("-w", "--workspace", help='name of workspace to use (i.e. "RhSn2")')
parser.add_argument("-f", "--file", help='Name of single file to clean.')
parser.add_argument('-s', '--skip-cleaned', help='Skip already cleaned datasets.', action="store_true")
args = parser.parse_args()

if arpes.config.CONFIG['WORKSPACE'] is None:
    arpes.config.CONFIG['WORKSPACE'] = args.workspace or os.getenv('WORKSPACE')

if arpes.config.CONFIG['WORKSPACE'] is None:
    raise ConfigurationError('You must provide a workspace.')


def is_clean(f, path=None):
    return 'cleaned' in f or (isinstance(path, str) and 'cleaned' in path)


if args.file:
    if args.skip_cleaned and cleaned_dataset_exists(args.file):
        print('Skipping cleaned')
    else:
        clean_xlsx_dataset(os.path.join(os.getcwd(), args.file), with_inferred_cols=False, reload=True, warn_on_exists=True,
                           header=args.offset - 1, allow_soft_match=True)
else:
    for path, _, files in walk(os.getcwd()):
        excel_files = [f for f in files if '.xlsx' in f or '.xlx'in f]

        for x in excel_files:
            print(x)
            if is_clean(x, path):
                continue

            if args.skip_cleaned and cleaned_dataset_exists(x):
                print('Skipping previously cleaned dataset.')
                continue

            clean_xlsx_dataset(os.path.join(path, x), with_inferred_cols=False, reload=True, warn_on_exists=True,
                               header=args.offset - 1, allow_soft_match=True)