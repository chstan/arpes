"""Provides a convenience data loader for example data.

Providing example data is essential for ensuring approachability,
but in the past we only provided a single ARPES cut. We now provide
a variety but need to be parsimonious about disk space for downloads.
As a result, this custom loader let's us pretend we store the data in
a higher quality format.
"""

import xarray as xr
from arpes.endstations import SingleFileEndstation, HemisphericalEndstation
import arpes.xarray_extensions

__all__ = ["ExampleDataEndstation"]


class ExampleDataEndstation(SingleFileEndstation, HemisphericalEndstation):
    """Loads data from exported .nc format saved by xarray. Used for storing example data."""

    PRINCIPAL_NAME = "example_data"

    _TOLERATED_EXTENSIONS = {".nc"}

    def load_single_frame(
        self, frame_path: str = None, scan_desc: dict = None, **kwargs
    ) -> xr.Dataset:
        """Loads single file examples.

        Additionally, copies coordinate offsets onto the dataset because we have
        preloaded these for convenience on maps.
        """
        data = xr.open_dataarray(frame_path)
        dataset = xr.Dataset({"spectrum": data})
        dataset.S.apply_offsets(data.S.offsets)

        return dataset
