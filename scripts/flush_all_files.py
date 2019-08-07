#!/usr/bin/env python

import os
import sys

from arpes.io import flush_cache
from arpes.utilities import walk_scans

sys.path.append('/Users/chstansbury/PyCharmProjects/python-arpes/')


to_remove = set()
for scan in walk_scans(os.getcwd(), only_id=True):
    to_remove.add(scan)

flush_cache(to_remove, True)
