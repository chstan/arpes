"""Implements a simple loader for Igor files.

This does not load data according to the PyARPES data model, so you should
ideally use a specific data loader where it is available.
"""
import xarray as xr
from arpes.endstations import (
    SingleFileEndstation,
)
from arpes.load_pxt import read_single_pxt

__all__ = ("IgorEndstation",)


class IgorEndstation(SingleFileEndstation):
    """A generic file loader for PXT files.

    This makes no assumptions about whether data is from a hemisphere
    or otherwise, so it might not be perfect for all Igor users, but it
    is a place to start and to demonstrate how to implement a data loading
    plugin.
    """

    PRINCIPAL_NAME = "Igor"
    ALIASES = [
        "IGOR",
        "pxt",
        "pxp",
        "Wave",
        "wave",
    ]

    _TOLERATED_EXTENSIONS = {
        ".pxt",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )

    RENAME_KEYS = {}

    MERGE_ATTRS = {}

    ATTR_TRANSFORMS = {}

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """Igor .pxt and .ibws are single files so we just read the one passed here."""
        print(frame_path, scan_desc)

        pxt_data = read_single_pxt(frame_path)
        return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)
