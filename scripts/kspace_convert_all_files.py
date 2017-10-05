#!/usr/bin/env python

import os
import sys

sys.path.append('/Users/chstansbury/PyCharmProjects/python-arpes/')

from arpes.utilities import walk_scans
from arpes.pipelines import convert_scan_to_kspace

for scan in walk_scans(os.getcwd(), only_id=True):
    print(scan)
    convert_scan_to_kspace(scan)