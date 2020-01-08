from PyQt5 import QtGui
import pyqtgraph as pg
import numpy as np
import typing

from collections import defaultdict, namedtuple

from arpes.utilities.ui import CursorRegion
from .data_array_image_view import DataArrayImageView, DataArrayPlot

import arpes.config

__all__ = ('SimpleApp',)

ReactivePlotRecord = namedtuple('ReactivePlotRecord', ('dims', 'view', 'orientation',))


class SimpleApp:
    """
    Has all of the layout information and business logic for an interactive data browsing utility using PyQt5.
    """

    WINDOW_CLS = None
    WINDOW_SIZE = (4,4,)
    TITLE = 'Untitled Tool'

    DEFAULT_COLORMAP = 'viridis'

    def __init__(self):
        self._ninety_eight_percentile = None
        self.data = None
        self.window = None
        self.settings = None
        self._layout = None

        self.context = {}

        self.views = {}
        self.reactive_views = []
        self.registered_cursors: typing.Dict[typing.List[CursorRegion]] = defaultdict(list)

        self.settings = arpes.config.SETTINGS.copy()

    @property
    def ninety_eight_percentile(self):
        if self._ninety_eight_percentile is not None:
            return self._ninety_eight_percentile

        self._ninety_eight_percentile = np.percentile(self.data.values, (98,))[0]
        return self._ninety_eight_percentile

    def print(self, *args, **kwargs):
        self.window.window_print(*args, **kwargs)

    @staticmethod
    def build_pg_cmap(colormap):
        sampled_colormap = colormap.colors[::51]
        return pg.ColorMap(pos=np.linspace(0, 1, len(sampled_colormap)), color=sampled_colormap)

    def set_colormap(self, colormap):
        import matplotlib.cm

        if isinstance(colormap, str):
            colormap = matplotlib.cm.get_cmap(colormap)

        cmap = self.build_pg_cmap(colormap)
        for view_name, view in self.views.items():
            if isinstance(view, DataArrayImageView):
                view.setColorMap(cmap)

    def generate_marginal_for(self, dimensions, column, row, name=None, orientation='horiz', cursors=False, layout=None):
        if layout is None:
            layout = self._layout

        remaining_dims = [l for l in list(range(len(self.data.dims))) if l not in dimensions]

        if len(remaining_dims) == 1:
            widget = DataArrayPlot(root=self, name=name, orientation=orientation)
            #widget = pg.PlotWidget(name=name)
            self.views[name] = widget

            if orientation == 'horiz':
                widget.setMaximumHeight(200)
            else:
                widget.setMaximumWidth(200)

            if cursors:
                cursor = CursorRegion(
                    orientation=CursorRegion.Vertical if orientation == 'vert' else CursorRegion.Horizontal,
                    movable=True)
                widget.addItem(cursor, ignoreBounds=False)
                self.connect_cursor(remaining_dims[0], cursor)
        else:
            assert len(remaining_dims) == 2
            widget = DataArrayImageView(name=name, root=self)
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
            ReactivePlotRecord(dims=dimensions, view=widget, orientation=orientation))
        layout.addWidget(widget, column, row)
        return widget

    def before_show(self):
        pass

    def after_show(self):
        pass

    def layout(self):
        raise NotImplementedError()

    def start(self):
        app = QtGui.QApplication([])

        from arpes.utilities.qt import qt_info
        qt_info.init_from_app(app)

        self.window = self.WINDOW_CLS()
        self.window.resize(*qt_info.inches_to_px(self.WINDOW_SIZE))
        self.window.setWindowTitle(self.TITLE)

        cw = QtGui.QWidget()
        self.window.setCentralWidget(cw)
        self.window.app = self

        self._layout = self.layout()
        self.before_show()
        if self.DEFAULT_COLORMAP is not None:
            self.set_colormap(self.DEFAULT_COLORMAP)

        cw.setLayout(self._layout)
        self.window.show()
        self.after_show()

        QtGui.QApplication.instance().exec()
