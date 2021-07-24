"""Performs some basic sanity checks during the packaging sequence.

This is done to verify that the conda-build didn't go abysmally wrong. 
This seems to happen occasionally for configuration reasons I don't 
entirely understand.
"""


def check_load_example_data():
    """Validate that we included resource data correctly."""
    import arpes.io
    import xarray as xr

    data = arpes.io.load_example_data()
    assert isinstance(data, xr.Dataset)


if __name__ == "__main__":
    check_load_example_data()
