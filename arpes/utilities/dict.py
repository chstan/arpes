"""Utilities for modifying, iterating over, and transforming dictionaries."""
import re
from collections import OrderedDict

from arpes.utilities.xarray import lift_dataarray_attrs, lift_datavar_attrs
from typing import Any, Dict, Union

__all__ = (
    "rename_keys",
    "clean_keys",
    "rename_dataarray_attrs",
    "clean_datavar_attribute_names",
    "clean_attribute_names",
    "case_insensitive_get",
)


def _rename_key(d: Union[Dict[str, Any], OrderedDict], k: str, nk: str) -> None:
    if k in d:
        d[nk] = d[k]
        del d[k]


def rename_keys(
    d: Union[Dict[str, Any], OrderedDict], keys_dict: Dict[str, str]
) -> Union[Dict[str, Any], OrderedDict]:
    """Renames all the keys of `d` according to the remapping in `keys_dict`."""
    d = d.copy()
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d


def clean_keys(d):
    """Renames dictionary keys so that they are more Pythonic."""

    def clean_single_key(k):
        k = k.replace(" ", "_")
        k = k.replace(".", "_")
        k = k.lower()
        k = re.sub(r"[()/?]", "", k)
        k = k.replace("__", "_")
        return k

    return dict(zip([clean_single_key(k) for k in d.keys()], d.values()))


def case_insensitive_get(d: dict, key: str, default=None, take_first=False):
    """Looks up a key in a dictionary ignoring case.

    We use this sometimes to be nicer to users who don't provide perfectly sanitized data.

    Args:
        d: The dictionary to perform lookup in
        key: The key to get
        default: A default value if the key is not present
        take_first: Whether to take the first entry if there were multiple found
    """
    found_value = False
    value = None

    for k, v in d.items():
        if k.lower() == key.lower():
            if not take_first and found_value:
                raise ValueError("Duplicate case insensitive keys")

            value = v
            found_value = True

            if take_first:
                break

    if not found_value:
        return default

    return value


rename_dataarray_attrs = lift_dataarray_attrs(rename_keys)
clean_attribute_names = lift_dataarray_attrs(clean_keys)

clean_datavar_attribute_names = lift_datavar_attrs(clean_keys)
