#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

sys.path.append(os.getenv('ARPES_ROOT'))
sys.path.append(os.path.join(os.getenv('ARPES_ROOT'), 'arpes'))

from arpes.utilities import modern_clean_xlsx_dataset
from arpes.utilities.dataset import walk_datasets
import arpes.config


DESCRIPTION = """
Command line tool to transform user readable dataset spreadsheets into ones that are automatically parseable
by the analysis code. This will probably work without arguments.
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("-w", "--workspace", help='name of workspace to use (i.e. "RhSn2"). '
                                              'This can usually be safely omitted.')
parser.add_argument("-f", "--file", help='Name of single file to clean.')

args = parser.parse_args()

arpes.config.attempt_determine_workspace(args.workspace or os.getenv('WORKSPACE'))

if args.file:
    print("├{}".format(args.file))
    files = [os.path.join(os.getcwd(args.file))]
else:
    files = walk_datasets()

for file in files:
    dataset = modern_clean_xlsx_dataset(file, with_inferred_cols=False, allow_soft_match=True, write=True)
