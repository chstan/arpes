#!/usr/bin/env python

import os
import sys

from arpes.pipelines import convert_scan_to_kspace
from arpes.utilities import walk_scans

sys.path.append('/Users/chstansbury/PyCharmProjects/python-arpes/')


for scan in walk_scans(os.getcwd(), only_id=True):
    print(scan)
    convert_scan_to_kspace(scan)
