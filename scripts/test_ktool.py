from arpes.io import load_data
from arpes.plotting.qt_ktool import ktool

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

data = load_data(str(data_path.absolute()), location="ALG-MC")

ktool(data)
