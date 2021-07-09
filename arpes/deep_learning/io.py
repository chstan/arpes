"""Facilitates saving intermediate data using a portable binary format for numpy."""
import json
from pathlib import Path

import numpy as np

__all__ = [
    "to_portable_bin",
    "from_portable_bin",
]

DTYPES = {
    "float64": np.float64,
    "float32": np.float32,
    "int64": np.int64,
    "int32": np.int32,
}


def from_portable_bin(path: Path) -> np.ndarray:
    """Reads data from a relatively portable binary format.

    A "portable" binary file is a directory containing
    a binary file and a very small json file which contains

    1. dtype
    2. shape
    3. offset

    We do this instead of using pickling in order to ensure that the
    data formats are portable.
    """
    with open(str(path / "portability.json"), "r") as f:
        portability = json.load(f)

    dtype = DTYPES[portability.pop("dtype")]
    shape = portability["shape"]

    arr = np.fromfile(str(path / "arr.bin"), dtype=dtype)
    arr = arr.reshape(shape)
    return arr


def to_portable_bin(arr: np.ndarray, path: Path) -> None:
    """Converts data to a relatively portable binary format.

    See also `read_portable_bin`.

    Writes array as a binary format with an associated json description
    of necessary portability info.
    """
    path.mkdir(parents=True, exist_ok=True)
    json_path, arr_path = path / "portability.json", path / "arr.bin"
    assert not (json_path.exists() or arr_path.exists())

    with open(str(json_path), "w") as f:
        json.dump(
            {
                "dtype": arr.dtype.name,
                "shape": arr.shape,
            },
            f,
        )

    arr.tofile(str(arr_path.resolve()))
