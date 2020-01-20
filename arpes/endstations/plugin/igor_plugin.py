import re
from pathlib import Path

import numpy as np

import typing
import xarray as xr
from arpes.endstations import (SingleFileEndstation,)
from arpes.load_pxt import read_single_pxt

__all__ = ('IgorEndstation',)


class IgorEndstation(SingleFileEndstation):
    """
    A generic file loader for PXT files. This makes no assumptions
    about whether data is from a hemisphere or otherwise, so it might not
    be perfect for all Igor users, but it is a place to start.
    """

    PRINCIPAL_NAME = 'Igor'
    ALIASES = ['IGOR', 'pxt', 'pxp', 'Wave', 'wave', ]

    _TOLERATED_EXTENSIONS = {'.pxt',}
    _SEARCH_PATTERNS = (
        r'[\-a-zA-Z0-9_\w]+_[0]+{}$',
        r'[\-a-zA-Z0-9_\w]+_{}$',
        r'[\-a-zA-Z0-9_\w]+{}$',
        r'[\-a-zA-Z0-9_\w]+[0]{}$',
    )

    RENAME_KEYS = {}

    MERGE_ATTRS = {}

    ATTR_TRANSFORMS = {}

    def load_single_frame(self, frame_path: str=None, scan_desc: dict = None, **kwargs):
        print(frame_path, scan_desc)

        pxt_data = read_single_pxt(frame_path)
        return xr.Dataset({'spectrum': pxt_data},attrs=pxt_data.attrs)