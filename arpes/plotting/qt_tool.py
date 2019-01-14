from PyQt5 import QtGui, QtCore
import PyQt5.QtGui
import pyqtgraph as pg
import numpy as np
from collections import defaultdict, namedtuple

from arpes.utilities import normalize_to_spectrum
from arpes.typing import DataType
import arpes.config

__all__ = ('QtTool', 'qt_tool',)

ReactivePlotRecord = namedtuple('ReactivePlotRecord', ('dims', 'view', 'orientation',))

class QtTool(object):
    """
    QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PyQt5 for now we retain a number of the
    metaphors from BokehTool, including a "context" that stores the state, and can be used to programmatically interface
    with the tool
    """
    def __init__(self):
        self.settings = None
        self.context = {}
        self.data = None
        self.ninety_eight_percentile = None
        self.layout = None
        self.views = {}
        self.reactive_views = []
        self.registered_cursors = defaultdict(list)

        self.settings = arpes.config.SETTINGS.copy()

    def configure_image_widgets(self):
        if len(self.data.dims) == 2:
            self.generate_marginal_for((), 1, 0, 'xy', cursors=True)
            self.generate_marginal_for((1,), 0, 0, 'x', orientation='horiz')
            self.generate_marginal_for((0,), 1, 1, 'y', orientation='vert')

        if len(self.data.dims) == 3:
            self.generate_marginal_for((1, 2), 0, 0, 'x', orientation='horiz')
            self.generate_marginal_for((1,), 1, 0, 'xz')
            self.generate_marginal_for((2,), 2, 0, 'xy', cursors=True)
            self.generate_marginal_for((0, 1,), 0, 1, 'z', orientation='horiz', cursors=True)
            self.generate_marginal_for((0, 2,), 2, 2, 'y', orientation='vert')
            self.generate_marginal_for((0,), 2, 1, 'yz')

        if len(self.data.dims) == 4:
            self.generate_marginal_for((1, 3), 0, 0, 'xz')
            self.generate_marginal_for((2, 3), 1, 0, 'xy', cursors=True)
            self.generate_marginal_for((0, 2,), 1, 1, 'yz')
            self.generate_marginal_for((0, 1,), 0, 1, 'zw', cursors=True)

    def set_colormap(self, colormap):
        sampled_colormap = colormap.colors[::51]
        cmap = pg.ColorMap(pos=np.linspace(0, 1, len(sampled_colormap)), color=sampled_colormap)
        for view_name, view in self.views.items():
            if isinstance(view, pg.ImageView):
                view.setColorMap(cmap)

    def connect_cursor(self, dimension, the_line):
        self.registered_cursors[dimension].append(the_line)

        def connected_cursor(line: pg.InfiniteLine):
            new_cursor = list(self.context['cursor'])
            new_cursor[dimension] = line.value()
            self.update_cursor_position(new_cursor)

        the_line.sigDragged.connect(connected_cursor)

    def update_cursor_position(self, new_cursor, force=False):
        old_cursor = list(self.context['cursor'])
        self.context['cursor'] = new_cursor

        changed_dimensions = [i for i, (x, y) in enumerate(zip(old_cursor, new_cursor)) if x != y]

        cursor_text = ','.join('{}: {:.4g}'.format(x, y) for x, y in zip(self.data.dims, new_cursor))
        self.window.statusBar().showMessage('({})'.format(cursor_text))

        # update data
        for reactive  in self.reactive_views:
            if len(set(reactive.dims).intersection(set(changed_dimensions))) or force:
                try:
                    select_coord = dict(zip([self.data.dims[i] for i in reactive.dims],
                                            [int(new_cursor[i]) for i in reactive.dims]))
                    if isinstance(reactive.view, pg.ImageView):
                        reactive.view.setImage(self.data.isel(**select_coord).values)
                    elif isinstance(reactive.view, pg.PlotWidget):
                        for_plot = self.data.isel(**select_coord).values
                        cursors = [l for l in reactive.view.getPlotItem().items if isinstance(l, pg.InfiniteLine)]
                        reactive.view.clear()
                        for c in cursors:
                            reactive.view.addItem(c)

                        if reactive.orientation == 'horiz':
                            reactive.view.plot(for_plot)
                        else:
                            reactive.view.plot(for_plot, range(len(for_plot)))
                except IndexError:
                    pass


    def generate_marginal_for(self, dimensions, column, row, name=None, orientation='horiz', cursors=False):
        remaining_dims = [l for l in list(range(len(self.data.dims))) if l not in dimensions]
        if len(remaining_dims) == 1:
            widget = pg.PlotWidget(name=name)
            self.views[name] = widget
            if orientation == 'horiz':
                widget.setMaximumHeight(200)
            else:
                widget.setMaximumWidth(200)

            if cursors:
                cursor = pg.InfiniteLine(angle=0 if orientation == 'vert' else 90, movable=True)
                widget.addItem(cursor, ignoreBounds=False)
                self.connect_cursor(remaining_dims[0], cursor)
        else:
            assert(len(remaining_dims) == 2)
            widget = pg.ImageView(name=name)
            widget.view.setAspectLocked(False)
            self.views[name] = widget

            widget.setHistogramRange(0, self.ninety_eight_percentile)
            widget.setLevels(0.05, 0.95)

            if cursors:
                cursor_vert = pg.InfiniteLine(angle=90, movable=True)
                cursor_horiz = pg.InfiniteLine(angle=0, movable=True)
                widget.addItem(cursor_vert, ignoreBounds=True)
                widget.addItem(cursor_horiz, ignoreBounds=True)
                self.connect_cursor(remaining_dims[0], cursor_vert)
                self.connect_cursor(remaining_dims[1], cursor_horiz)

        self.reactive_views.append(
            ReactivePlotRecord(dims=dimensions, view=widget, orientation=orientation))
        self.layout.addWidget(widget, column, row)
        return widget

    def start(self):
        app = QtGui.QApplication([])

        win = QtGui.QMainWindow()
        win.resize(1100, 1100)
        win.setWindowTitle('QT Tool')
        cw = QtGui.QWidget()
        win.setCentralWidget(cw)
        self.window = win
        self.layout = QtGui.QGridLayout()

        cw.setLayout(self.layout)

        # add main image widgets
        self.configure_image_widgets()
        import matplotlib.cm
        self.set_colormap(matplotlib.cm.viridis)
        win.show()

        # basic state initialization
        self.context.update({
            'cursor': [self.data.coords[d].mean().item() for d in self.data.dims],
        })

        ## Display the data
        self.update_cursor_position(self.context['cursor'], force=True)

        QtGui.QApplication.instance().exec()

    def set_data(self, data: DataType):
        data = normalize_to_spectrum(data)
        self.ninety_eight_percentile = np.percentile(data.values, (98,))[0]
        self.data = data


def qt_tool(data: DataType):
    tool = QtTool()
    tool.set_data(data)
    tool.start()