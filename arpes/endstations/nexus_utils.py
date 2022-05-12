"""Provides a jumping off point for defining data loading plugins using the NeXuS file format.

Currently we assume that the raw file format is actually HDF.
"""

import numpy as np
import h5py as h5
import xarray as xr
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable

__all__ = ["read_data_attributes_from"]


def read_group_data(group, attribute_name=None) -> Any:
    if attribute_name is not None:
        try:
            data = group[attribute_name]["data"]
        except ValueError:
            data = group[attribute_name]
    else:
        try:
            data = group["data"]
        except ValueError:
            data = group

    try:
        data = data[:]
    except ValueError:
        data = data[()]

    if isinstance(data, np.ndarray):
        try:
            data = data.item()
        except ValueError:
            pass

    return data


@dataclass
class Target:
    name: str = ""
    read: Callable = field(default=lambda x: x)

    value: Any = None

    def read_h5(self, g, path):
        self.value = None
        self.value = self.read(read_group_data(g))

    def write_to_dataarray(self, arr: xr.DataArray):
        pass

    def write_to_dataset(self, dset: xr.Dataset):
        pass


@dataclass
class DebugTarget(Target):
    name = "debug"

    def read_h5(self, g, path):
        print(path, self.read(read_group_data(g)))


@dataclass
class AttrTarget(Target):
    def write_to_dataarray(self, arr: xr.DataArray):
        arr.attrs[self.name] = self.value


@dataclass
class CoordTarget(Target):
    def write_to_dataarray(self, arr: xr.DataArray):
        arr.coords[self.name] = self.value


def read_data_attributes_from_tree(group, tree, targets=None, path=None) -> List[Target]:
    """Reads simple (float, string, etc.) leaves from nested paths out of a NeXuS file.

    This is handled in a more robust way because we use two stages
    to (1) read the attribute data from the file and (2) bind the data
    we read to the target xr.DataArray and xr.Dataset.

    Args:
        group: The NeXuS/HDF Group or File object to read from
        tree: A binding tree whose leaves are `Target` instances

    Returns:
        Flat collection of `Target` instances which have been populated
    """

    if targets is None:
        targets = []

    if path is None:
        path = []

    if isinstance(tree, Target):
        tree = [tree]

    if isinstance(tree, (list, tuple)):
        for t in tree:
            t.read_h5(group, path)
            targets.append(t)

        return targets

    marked = set()
    for k, g in group.items():
        if k in tree:
            marked.add(k)
            path.append(k)
            read_data_attributes_from_tree(g, tree[k], targets, path)
            path.pop()

    for k, g in group.attrs.items():
        if k in marked:
            raise ValueError(f"Already encountered {k}. Skipping")
        if k in tree and k not in marked:
            path.append(k)
            read_data_attributes_from_tree(g, tree[k], targets, path)
            path.pop()

    return targets


def read_data_attributes_from(group, paths) -> Dict[str, Any]:
    """Reads simple (float, string, etc.) leaves from nested paths out of a NeXuS file.

    Args:
        group: The NeXuS/HDF group object to read from
        paths: The paths to collect leaves from

    Returns:
        Flat collection of attributes
    """
    read_attrs = {}
    original_group = group
    for path, attributes in paths:
        group = original_group
        for p in path:
            group = group[p]

        for attribute_name in attributes:
            data = read_group_data(group, attribute_name)
            read_attrs[attribute_name] = data

    return read_attrs
