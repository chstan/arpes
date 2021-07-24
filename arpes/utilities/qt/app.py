"""Application infrastructure for apps/tools which browse a data volume."""
from PyQt5 import QtGui
import pyqtgraph as pg
import numpy as np
import typing
import xarray as xr
import weakref

from collections import defaultdict

from arpes.utilities.ui import CursorRegion
from .data_array_image_view import DataArrayImageView, DataArrayPlot
from .utils import PlotOrientation, ReactivePlotRecord

import arpes.config

__all__ = ["SimpleApp"]


class SimpleApp:
    """Has all of the layout information and business logic for an interactive data browsing utility using PyQt5."""

    WINDOW_CLS = None
    WINDOW_SIZE = (4, 4)
    TITLE = "Untitled Tool"

    DEFAULT_COLORMAP = "viridis"

    _data = None

    def __init__(self):
        """Only interesting thing on init is to make a copy of the user settings."""
        self._ninety_eight_percentile = None
        self._data = None
        self.settings = None
        self._window = None
        self._layout = None

        self.context = {}

        self.views = {}
        self.reactive_views = []
        self.registered_cursors: typing.Dict[typing.List[CursorRegion]] = defaultdict(list)

        self.settings = arpes.config.SETTINGS.copy()

    def copy_to_clipboard(self, value: typing.Any) -> None:
        """Attempts to copy the value to the clipboard, or else prints."""
        try:
            import pyperclip
            import pprint

            pyperclip.copy(pprint.pformat(value))
        except ImportError:
            pass
        finally:
            import pprint

            print(pprint.pformat(value))

    @property
    def data(self) -> xr.DataArray:
        """Read data from the cached attribute.

        This is a propety as opposed to a plain attribute
        in order to facilitate rendering datasets with several
        data_vars.
        """
        return self._data

    @data.setter
    def data(self, new_data: xr.DataArray):
        self._data = new_data

    def close(self):
        """Graceful shutdown. Tell each view to close and drop references so GC happens."""
        for v in self.views.values():
            v.close()

        self.views = {}
        self.reactive_views = []

    @property
    def ninety_eight_percentile(self):
        """Calculates the 98 percentile of data so colorscale is not outlier dependent."""
        if self._ninety_eight_percentile is not None:
            return self._ninety_eight_percentile

        self._ninety_eight_percentile = np.percentile(self.data.values, (98,))[0]
        return self._ninety_eight_percentile

    def print(self, *args, **kwargs):
        """Forwards printing to the application so it ends up in Jupyter."""
        self.window.window_print(*args, **kwargs)

    @staticmethod
    def build_pg_cmap(colormap):
        """Converts a matplotlib colormap to one suitable for pyqtgraph.

        pyqtgraph uses its own colormap format but for consistency and aesthetic
        reasons we want to use the ones from matplotlib. This will sample the colors
        from the colormap and convert it into an array suitable for pyqtgraph.
        """
        sampled_colormap = colormap.colors[::51]
        sampled_colormap = np.array([s + [1.0] for s in sampled_colormap])

        # need to scale colors if pyqtgraph is older.
        if pg.__version__.split(".")[1] != "10":
            sampled_colormap = sampled_colormap * 255  # super frustrating undocumented change

        return pg.ColorMap(pos=np.linspace(0, 1, len(sampled_colormap)), color=sampled_colormap)

    def set_colormap(self, colormap):
        """Finds all `DataArrayImageView` instances and sets their color palette."""
        import matplotlib.cm

        if isinstance(colormap, str):
            colormap = matplotlib.cm.get_cmap(colormap)

        cmap = self.build_pg_cmap(colormap)
        for view_name, view in self.views.items():
            if isinstance(view, DataArrayImageView):
                view.setColorMap(cmap)

    def generate_marginal_for(
        self,
        dimensions,
        column,
        row,
        name=None,
        orientation=PlotOrientation.Horizontal,
        cursors=False,
        layout=None,
    ):
        """Generates a marginal plot for this applications's data after selecting along `dimensions`.

        This is used to generate the many different views of a volume in the browsable tools.
        """
        if layout is None:
            layout = self._layout

        remaining_dims = [l for l in list(range(len(self.data.dims))) if l not in dimensions]

        if len(remaining_dims) == 1:
            widget = DataArrayPlot(name=name, root=weakref.ref(self), orientation=orientation)
            self.views[name] = widget

            if orientation == PlotOrientation.Horizontal:
                widget.setMaximumHeight(200)
            else:
                widget.setMaximumWidth(200)

            if cursors:
                cursor = CursorRegion(
                    orientation=CursorRegion.Vertical
                    if orientation == PlotOrientation.Vertical
                    else CursorRegion.Horizontal,
                    movable=True,
                )
                widget.addItem(cursor, ignoreBounds=False)
                self.connect_cursor(remaining_dims[0], cursor)
        else:
            assert len(remaining_dims) == 2
            widget = DataArrayImageView(name=name, root=weakref.ref(self))
            widget.view.setAspectLocked(False)
            self.views[name] = widget

            widget.setHistogramRange(0, self.ninety_eight_percentile)
            widget.setLevels(0.05, 0.95)

            if cursors:
                cursor_vert = CursorRegion(orientation=CursorRegion.Vertical, movable=True)
                cursor_horiz = CursorRegion(orientation=CursorRegion.Horizontal, movable=True)
                widget.addItem(cursor_vert, ignoreBounds=True)
                widget.addItem(cursor_horiz, ignoreBounds=True)
                self.connect_cursor(remaining_dims[0], cursor_vert)
                self.connect_cursor(remaining_dims[1], cursor_horiz)

        self.reactive_views.append(
            ReactivePlotRecord(dims=dimensions, view=widget, orientation=orientation)
        )
        layout.addWidget(widget, column, row)
        return widget

    def before_show(self):
        """Lifecycle hook."""
        pass

    def after_show(self):
        """Lifecycle hook."""
        pass

    def layout(self):
        """Hook for defining the application layout.

        This needs to be provided by subclasses.
        """
        raise NotImplementedError

    @property
    def window(self):
        """Gets the window instance on the current application."""
        return self._window

    def start(self):
        """Starts the Qt application, configures the window, and begins Qt execution."""
        # When running in nbconvert, don't actually open tools.
        import arpes.config

        if arpes.config.DOCS_BUILD:
            return

        app = QtGui.QApplication([])
        app.owner = self
        # self.app = app

        from arpes.utilities.qt import qt_info

        qt_info.init_from_app(app)

        self._window = self.WINDOW_CLS()
        self.window.resize(*qt_info.inches_to_px(self.WINDOW_SIZE))
        self.window.setWindowTitle(self.TITLE)

        self.cw = QtGui.QWidget()
        self._layout = self.layout()
        self.cw.setLayout(self._layout)
        self.window.setCentralWidget(self.cw)
        self.window.app = weakref.ref(self)

        self.before_show()

        if self.DEFAULT_COLORMAP is not None:
            self.set_colormap(self.DEFAULT_COLORMAP)

        self.window.show()

        self.after_show()
        qt_info.apply_settings_to_app(app)

        QtGui.QApplication.instance().exec()
