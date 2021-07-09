from arpes.io import load_data
from arpes.plotting.basic_tools import path_tool, mask_tool, bkg_tool

from pathlib import Path

data_path = (
    Path(__file__).parent.parent
    / "tests"
    / "resources"
    / "datasets"
    / "basic"
    / "data"
    / "main_chamber_cut_0.fits"
)

data = load_data(str(data_path.absolute()))

# data = xr.DataArray(np.random.random((100,100,100)))

bkg_tool(data)
