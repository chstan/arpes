from arpes.io import load_data
from arpes.plotting.qt_tool import qt_tool

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

qt_tool(data)
