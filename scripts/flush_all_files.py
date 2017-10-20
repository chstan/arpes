#!/usr/bin/env python

import os
import sys

sys.path.append('/Users/chstansbury/PyCharmProjects/python-arpes/')

from arpes.utilities import walk_scans
from arpes.io import flush_cache

to_remove = set()
for scan in walk_scans(os.getcwd(), only_id=True):
    to_remove.add(scan)

flush_cache(to_remove, True)