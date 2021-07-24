"""A live momentun conversion tool, useful for finding and setting offsets."""
from PyQt5 import QtWidgets
import numpy as np

from arpes.utilities import normalize_to_spectrum, group_by
from arpes.typing import DataType
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.qt import qt_info, SimpleApp, SimpleWindow
from arpes.utilities.ui import tabs, horizontal, vertical, label, numeric_input, CollectUI
from arpes.plotting.bz import segments_standard


__all__ = (
    "KTool",
    "ktool",
)


qt_info.setup_pyqtgraph()


class KTool(SimpleApp):
    """Provides a live momentum converting tool.

    QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PyQt5 for now we retain
    a number of the metaphors from BokehTool, including a "context" that stores the state, and can
    be used to programmatically interface with the tool.
    """

    TITLE = "KSpace-Tool"
    WINDOW_SIZE = (5, 6)
    WINDOW_CLS = SimpleWindow

    DEFAULT_COLORMAP = "viridis"

    def __init__(self, apply_offsets=True, zone=None, **kwargs):
        """Set attributes to safe defaults and unwrap the Brillouin zone definition."""
        super().__init__()

        if isinstance(
            zone,
            (tuple, list),
        ):
            self.segments_x, self.segments_y = zone
        elif zone:
            self.segments_x, self.segments_y = segments_standard(zone)
        else:
            self.segments_x, self.segments_y = None, None

        self.conversion_kwargs = kwargs
        self.data = None
        self.content_layout = None
        self.main_layout = None
        self.apply_offsets = apply_offsets

    def configure_image_widgets(self):
        """We have two marginals because we deal with Fermi surfaces, they get configured here."""
        self.generate_marginal_for((), 0, 0, "xy", cursors=False, layout=self.content_layout)
        self.generate_marginal_for((), 1, 0, "kxy", cursors=False, layout=self.content_layout)

    def add_contextual_widgets(self):
        """The main UI layout for controls and tools."""
        convert_dims = ["theta", "beta", "phi", "psi"]
        if "eV" not in self.data.dims:
            convert_dims += ["chi"]
        if "hv" in self.data.dims:
            convert_dims += ["hv"]

        ui = {}
        with CollectUI(ui):
            controls = tabs(
                [
                    "Controls",
                    horizontal(
                        *[
                            vertical(
                                *[
                                    vertical(
                                        label(p),
                                        numeric_input(
                                            self.data.attrs.get(f"{p}_offset", 0.0),
                                            input_type=float,
                                            id=f"control-{p}",
                                        ),
                                    )
                                    for p in pair
                                ]
                            )
                            for pair in group_by(2, convert_dims)
                        ]
                    ),
                ]
            )

        def update_dimension_name(dim_name):
            def updater(value):
                self.update_offsets(dict([[dim_name, float(value)]]))

            return updater

        for dim in convert_dims:
            ui[f"control-{dim}"].subject.subscribe(update_dimension_name(dim))

        controls.setFixedHeight(qt_info.inches_to_px(1.75))

        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(controls, 1, 0)

    def update_offsets(self, offsets):
        """Pushes offsets to the display data and optionally, the original data."""
        self.data.S.apply_offsets(offsets)
        if self.apply_offsets:
            self.original_data.S.apply_offsets(offsets)
        self.update_data()

    def layout(self):
        """Initialize the layout components."""
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        return self.main_layout

    def update_data(self):
        """The main redraw method for this tool.

        Converts data into momentum space and populates both the angle-space and momentum
        space views.

        If a Brillouin zone was requestd, plots that over the data as well.
        """
        self.views["xy"].setImage(self.data)

        kdata = convert_to_kspace(self.data, **self.conversion_kwargs)
        if "eV" in kdata.dims:
            kdata = kdata.S.transpose_to_back("eV")

        self.views["kxy"].setImage(kdata.S.nan_to_num())
        if self.segments_x is not None:
            bz_plot = self.views["kxy"].plot_item
            kx, ky = self.conversion_kwargs["kx"], self.conversion_kwargs["ky"]
            for segx, segy in zip(self.segments_x, self.segments_y):
                bz_plot.plot((segx - kx[0]) / (kx[1] - kx[0]), (segy - ky[0]) / (ky[1] - ky[0]))

    def before_show(self):
        """Lifecycle hook for configuration before app show."""
        self.configure_image_widgets()
        self.add_contextual_widgets()
        if self.DEFAULT_COLORMAP is not None:
            self.set_colormap(self.DEFAULT_COLORMAP)

    def after_show(self):
        """Initialize application state after app show. Just redraw."""
        self.update_data()

    def set_data(self, data: DataType):
        """Sets the current data to a new value and resets binning.

        Above what happens in QtTool, we try to extract a Fermi surface, and
        repopulate the conversion.
        """
        original_data = normalize_to_spectrum(data)
        self.original_data = original_data

        if len(data.dims) > 2:
            assert "eV" in original_data.dims
            data = data.sel(eV=slice(-0.05, 0.05)).sum("eV", keep_attrs=True)
            data.coords["eV"] = 0
        else:
            data = original_data

        if "eV" in data.dims:
            data = data.S.transpose_to_back("eV")

        self.data = data.copy(deep=True)

        if not self.conversion_kwargs:
            rng_mul = 1
            if data.coords["hv"] < 12:
                rng_mul = 0.5
            if data.coords["hv"] < 7:
                rng_mul = 0.25

            if "eV" in self.data.dims:
                self.conversion_kwargs = {
                    "kp": np.linspace(-2, 2, 400) * rng_mul,
                }
            else:
                self.conversion_kwargs = {
                    "kx": np.linspace(-2, 2, 300) * rng_mul,
                    "ky": np.linspace(-2, 2, 300) * rng_mul,
                }


def ktool(data: DataType, **kwargs):
    """Start the momentum conversion tool."""
    tool = KTool(**kwargs)
    tool.set_data(data)
    tool.start()
    return tool
