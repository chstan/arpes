"""Provides the core IO facilities supported by PyARPES.

The most important here are the data loading functions (load_data, load_example_data).
and pickling utilities.

Heavy lifting is actually performed by the plugin definitions which know how to ingest
different data formats into the PyARPES data model.

TODO: An improvement could be made to the example data if served
over a network and someone was willing to host a few larger pieces
of data.
"""

import pickle
import warnings

from typing import Any, List, Union, Optional
from dataclasses import dataclass

from pathlib import Path

import pandas as pd
import xarray as xr

import arpes.config
from arpes.endstations import load_scan
from arpes.typing import DataType

__all__ = (
    "load_data",
    "load_example_data",
    "easy_pickle",
    "list_pickles",
    "stitch",
)


def load_data(
    file: Union[str, Path, int], location: Optional[Union[str, type]] = None, **kwargs
) -> xr.Dataset:
    """Loads a piece of data using available plugins. This the user facing API for data loading.

    Args:
        file: An identifier for the file which should be loaded. If this is a number or can be coerced to one,
          data will be loaded from the workspace data folder if a matching unique file can be found for the number.
          If the value is a relative path, locations relative to the cwd and the workspace data folder will be checked.
          Absolute paths can also be used in a pinch.
        location: The name of the endstation/plugin to use. You should try to provide one. If None is provided,
          the loader will try to find an appropriate one based on the file extension and brute force. This will be slower
          and can be error prone in certain circumstances.

          Optionally, you can pass a loading plugin (the class) through this kwarg and directly specify
          the class to be used.


    Returns:
        The loaded data. Ideally, data which is loaded through the plugin system should be highly compliant with
        the PyARPES data model and should work seamlessly with PyARPES analysis code.
    """
    try:
        file = int(str(file))
    except ValueError:
        file = str(Path(file).absolute())

    desc = {
        "file": file,
        "location": location,
    }

    if location is None:
        desc.pop("location")
        warnings.warn(
            (
                "You should provide a location indicating the endstation or instrument used directly when "
                "loading data without a dataset. We are going to do our best but no guarantees."
            )
        )

    return load_scan(desc, **kwargs)


DATA_EXAMPLES = {
    "cut": ("ALG-MC", "cut.fits"),
    "map": ("example_data", "fermi_surface.nc"),
    "photon_energy": ("example_data", "photon_energy.nc"),
    "nano_xps": ("example_data", "nano_xps.nc"),
    "temperature_dependence": ("example_data", "temperature_dependence.nc"),
}


def load_example_data(example_name="cut") -> xr.Dataset:
    """Provides sample data for executable documentation."""
    if example_name not in DATA_EXAMPLES:
        warnings.warn(
            f"Could not find requested example_name: {example_name}. Please provide one of {list(DATA_EXAMPLES.keys())}"
        )

    location, example = DATA_EXAMPLES[example_name]
    file = Path(__file__).parent / "example_data" / example
    return load_data(file=file, location=location)


@dataclass
class ExampleData:
    @property
    def cut(self) -> xr.DataArray:
        return load_example_data("cut")

    @property
    def map(self) -> xr.DataArray:
        return load_example_data("map")

    @property
    def photon_energy(self) -> xr.DataArray:
        return load_example_data("photon_energy")

    @property
    def nano_xps(self) -> xr.DataArray:
        return load_example_data("nano_xps")

    @property
    def temperature_dependence(self) -> xr.DataArray:
        return load_example_data("temperature_dependence")


example_data = ExampleData()


def stitch(
    df_or_list: Union[List[str], pd.DataFrame],
    attr_or_axis: str,
    built_axis_name: Optional[str] = None,
    sort: bool = True,
) -> DataType:
    """Stitches together a sequence of scans or a DataFrame.

    Args:
        df_or_list: The list of the files to load
        attr_or_axis: Coordinate or attribute in order to promote to an index. I.e. if 't_a' is specified,
                      we will create a new axis corresponding to the temperature and concatenate the data along this axis
        built_axis_name: The name of the concatenated output dimensions
        sort: Whether to sort inputs to the concatenation according to their `attr_or_axis` value.

    Returns:
        The concatenated data.
    """
    list_of_files = None
    if isinstance(df_or_list, (pd.DataFrame,)):
        list_of_files = list(df_or_list.index)
    else:
        if not isinstance(df_or_list, (list, tuple)):
            raise TypeError("Expected an interable for a list of the scans to stitch together")

        list_of_files = list(df_or_list)

    if built_axis_name is None:
        built_axis_name = attr_or_axis

    if not list_of_files:
        raise ValueError("Must supply at least one file to stitch")

    loaded = [
        f if isinstance(f, (xr.DataArray, xr.Dataset)) else load_data(f) for f in list_of_files
    ]

    for i, loaded_file in enumerate(loaded):
        value = None
        if isinstance(attr_or_axis, (list, tuple)):
            value = attr_or_axis[i]
        elif attr_or_axis in loaded_file.attrs:
            value = loaded_file.attrs[attr_or_axis]
        elif attr_or_axis in loaded_file.coords:
            value = loaded_file.coords[attr_or_axis]

        loaded_file = loaded_file.assign_coords(dict([[built_axis_name, value]]))

    if sort:
        loaded.sort(key=lambda x: x.coords[built_axis_name])

    concatenated = xr.concat(loaded, dim=built_axis_name)
    if "id" in concatenated.attrs:
        del concatenated.attrs["id"]

    from arpes.provenance import provenance_multiple_parents

    provenance_multiple_parents(
        concatenated,
        loaded,
        {
            "what": "Stitched together separate datasets",
            "by": "stitch",
            "dim": built_axis_name,
        },
    )

    return concatenated


def file_for_pickle(name):
    here = Path(".")

    from arpes.config import CONFIG

    if CONFIG["WORKSPACE"]:
        here = Path(CONFIG["WORKSPACE"]["path"])

    path = here / "picklejar" / "{}.pickle".format(name)
    path.parent.mkdir(exist_ok=True)
    return str(path)


def load_pickle(name: str) -> Any:
    """Loads a workspace local pickle. Inverse to `save_pickle`."""
    with open(file_for_pickle(name), "rb") as file:
        return pickle.load(file)


def save_pickle(data: Any, name: str):
    """Saves a workspace local pickle. Inverse to `load_pickle`."""
    pickle.dump(data, open(file_for_pickle(name), "wb"))


def easy_pickle(data_or_str: Any, name=None) -> Any:
    """A convenience function around pickling.

    Provides a workspace scoped associative set of named pickles which
    can be used for

    Examples:
        Retaining analysis results between sessions.

        Sharing results between workspaces.

        Cacheing expensive or interim work.

    For reproducibility reasons, you should generally prefer to
    duplicate anaysis results using common code to prevent stale data
    dependencies, but there are good reasons to use pickling as well.

    This function knows whether we are pickling or unpickling depending on
    whether one or two arguments are provided.

    Args:
        data_or_str: If saving, the data to be pickled. If loading, the name of the pickle to load.
        name: If saving (non-None value), the name to associate. Defaults to None.

    Returns:
        None if name is not None, which indicates that we are saving data.
        Otherwise, returns the unpickled value associated to `name`.
    """
    # we are loading data
    if isinstance(data_or_str, str) or name is None:
        return load_pickle(data_or_str)

    # we are saving data
    assert isinstance(name, str)
    save_pickle(data_or_str, name)


def list_pickles() -> List[str]:
    """Generates a summary list of (workspace-local) pickled results and data.

    Returns:
        A list of the named pickles, suitable for passing to `easy_pickle`.
    """
    return [str(s.stem) for s in Path(file_for_pickle("just-a-pickle")).parent.glob("*.pickle")]
