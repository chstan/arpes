import sys
import traceback

from PyQt5 import QtGui, QtCore, QtWidgets
import PyQt5.QtGui
import pyqtgraph as pg
import numpy as np
from collections import namedtuple, defaultdict

from arpes.utilities import normalize_to_spectrum
from arpes.typing import DataType
import arpes.config

from .utils import PRETTY_KEYS, pretty_key_event, KeyBinding, hlayout, vlayout, layout, tabs
from .HelpDialog import HelpDialog
from .AxisInfoWidget import AxisInfoWidget
from .DataArrayImageView import DataArrayImageView
from .BinningInfoWidget import BinningInfoWidget

__all__ = ('QtTool', 'qt_tool',)

ReactivePlotRecord = namedtuple('ReactivePlotRecord', ('dims', 'view', 'orientation',))

pg.setConfigOptions(antialias=True, foreground=(0, 0, 0), background=(255, 255, 255))


def clamp(x, low, high):
    if x <= low:
        return low
    if x >= high:
        return high
    return x


class CursorRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._region_width = 1
        self.lines[1].setMovable(False)

    def set_width(self, value):
        self._region_width = value
        self.lineMoved(0)

    def lineMoved(self, i):
        if self.blockLineSignal:
            return

        self.lines[1].setValue(self.lines[0].value() + self._region_width)
        self.prepareGeometryChange()
        self.sigRegionChanged.emit(self)


def patchedLinkedViewChanged(self, view, axis):
    """
    This still isn't quite right but it is much better than before. For some reason
    the screen coordinates of the PlotWidget are not being computed correctly, so
    we will just lock them as though they were perfectly aligned.

    This will clearly not work well for plots that have to be coordinated across
    different parts of the layout, but this will work for now.

    We also don't handle inverted axes for now.
    :param self:
    :param view:
    :param axis:
    :return:
    """
    if self.linksBlocked or view is None:
        return

    vr = view.viewRect()
    vg = view.screenGeometry()
    sg = self.screenGeometry()
    if vg is None or sg is None:
        return

    view.blockLink(True)

    try:
        if axis == pg.ViewBox.XAxis:
            upp = float(vr.width()) / vg.width()
            overlap = min(sg.right(), vg.right()) - max(sg.left(), vg.left())

            if overlap < min(vg.width() / 3, sg.width() / 3):
                x1 = vr.left()
                x2 = vr.right()
            else: # attempt to align
                x1 = vr.left()
                x2 = vr.right() + (sg.width() - vg.width()) * upp

            self.enableAutoRange(pg.ViewBox.XAxis, False)
            self.setXRange(x1, x2, padding=0)
        else:
            upp = float(vr.height()) / vg.height()
            overlap = min(sg.bottom(), vg.bottom()) - max(sg.top(), vg.top())

            if overlap < min(vg.height() / 3, sg.height() / 3):
                y1 = vr.top()
                y2 = vr.bottom()
            else: # again, attempt to align
                y1 = vr.top() # snap them at one side to the same coordinate

                # and scale the other side
                y2 = vr.bottom() + (sg.height() - vg.height()) * upp

            self.enableAutoRange(pg.ViewBox.YAxis, False)
            self.setYRange(y1, y2, padding=0)
    finally:
        view.blockLink(False)


pg.ViewBox.linkedViewChanged = patchedLinkedViewChanged


def patched_excepthook(exc_type, exc_value, exc_tb):
    enriched_tb = _add_missing_frames(exc_tb) if exc_tb else exc_tb
    traceback.print_exception(exc_type, exc_value, enriched_tb)

def _add_missing_frames(tb):
    result = fake_tb(tb.tb_frame, tb.tb_lasti, tb.tb_lineno, tb.tb_next)
    frame = tb.tb_frame.f_back
    while frame:
        result = fake_tb(frame, frame.f_lasti, frame.f_lineno, result)
        frame = frame.f_back
    return result

fake_tb = namedtuple('fake_tb', ('tb_frame', 'tb_lasti', 'tb_lineno', 'tb_next'))


class QtToolWindow(QtGui.QMainWindow, QtCore.QObject):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tool = None
        self._help_dialog = None
        self._old_excepthook = sys.excepthook
        sys.excepthook = patched_excepthook

        self._keyBindings = [
            KeyBinding('Close', [QtCore.Qt.Key_Escape], self.do_close),
            KeyBinding('Toggle Help', [QtCore.Qt.Key_H], self.toggle_help),
            KeyBinding(
                'Scroll Cursor',
                [QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down],
                self.scroll
            ),
        ]

        QtGui.QGuiApplication.installEventFilter(self, self)

    def close(self):
        sys.excepthook = self._old_excepthook
        super(QtToolWindow, self).close()

    def do_close(self, event):
        self.close()

    def toggle_help(self, event):
        if self._help_dialog is None:
            self._help_dialog = HelpDialog(shortcuts=self._keyBindings)
            self._help_dialog.show()
            self._help_dialog._main_window = self
        else:
            self._help_dialog.close()
            self._help_dialog = None

    def window_print(self, *args, **kwargs):
        print(*args, **kwargs)

    def scroll(self, event):
        key_map = {
            QtCore.Qt.Key_Left: (0, -1),
            QtCore.Qt.Key_Right: (0, 1),
            QtCore.Qt.Key_Down: (1, -1),
            QtCore.Qt.Key_Up: (1, 1),
        }

        delta = key_map.get(event.key())
        if delta is not None and self.tool is not None:
            cursor = list(self.tool.context['cursor'])
            cursor[delta[0]] += delta[1]

            self.tool.update_cursor_position(cursor)

    def eventFilter(self, source, event):
        special_keys = [QtCore.Qt.Key_Down, QtCore.Qt.Key_Up, QtCore.Qt.Key_Left, QtCore.Qt.Key_Right]

        if event.type() in [QtCore.QEvent.KeyPress, QtCore.QEvent.ShortcutOverride]:
            if event.type() != QtCore.QEvent.ShortcutOverride or event.key() in special_keys:
                self.handleKeyPressEvent(event)

        return super().eventFilter(source, event)

    def handleKeyPressEvent(self, event):
        handled = False
        for binding in self._keyBindings:
            for combination in binding.chord:
                # only detect single keypresses for now
                if combination == event.key():
                    handled = True
                    binding.handler(event)

        if not handled:
            if arpes.config.SETTINGS.get('DEBUG', False):
                print(event.key())


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
        self.content_layout = None
        self.main_layout = None
        self.views = {}
        self.reactive_views = []
        self.registered_cursors = defaultdict(list)

        self.axis_info_widgets = []
        self.binning_info_widgets = []

        self.axes_tab = None
        self.binning_tab = None
        self._binning = None

        self.settings = arpes.config.SETTINGS.copy()

    @property
    def binning(self):
        if self._binning is None:
            return [1 for _ in self.data.dims]

        return list(self._binning)

    @binning.setter
    def binning(self, value):
        different_binnings = [i for i, (nv, v) in enumerate(zip(value, self._binning)) if nv != v]
        self._binning = value

        for i in different_binnings:
            cursors = self.registered_cursors.get(i)
            for cursor in cursors:
                cursor.set_width(self._binning[i])

        self.update_cursor_position(self.context['cursor'], force=True)

    def print(self, *args, **kwargs):
        self.window.window_print(*args, **kwargs)

    def transpose(self, transpose_order):
        reindex_order = [self.data.dims.index(t) for t in transpose_order]
        self.data = self.data.transpose(*transpose_order)

        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        self.update_cursor_position([self.context['cursor'][i] for i in reindex_order], force=True)

    def transpose_to_front(self, dim):
        if not isinstance(dim, str):
            dim = self.data.dims[dim]

        order = list(self.data.dims)
        order.remove(dim)
        order = [dim] + order
        self.transpose(order)

        # TODO update cursor, marginals, data

    def configure_image_widgets(self):
        if len(self.data.dims) == 2:
            self.generate_marginal_for((), 1, 0, 'xy', cursors=True)
            self.generate_marginal_for((1,), 0, 0, 'x', orientation='horiz')
            self.generate_marginal_for((0,), 1, 1, 'y', orientation='vert')

            self.views['xy'].view.setYLink(self.views['y'])
            self.views['xy'].view.setXLink(self.views['x'])

        if len(self.data.dims) == 3:
            self.generate_marginal_for((1, 2), 0, 0, 'x', orientation='horiz')
            self.generate_marginal_for((1,), 1, 0, 'xz')
            self.generate_marginal_for((2,), 2, 0, 'xy', cursors=True)
            self.generate_marginal_for((0, 1,), 0, 1, 'z', orientation='horiz', cursors=True)
            self.generate_marginal_for((0, 2,), 2, 2, 'y', orientation='vert')
            self.generate_marginal_for((0,), 2, 1, 'yz')

            self.views['xy'].view.setYLink(self.views['y'])
            self.views['xy'].view.setXLink(self.views['x'])
            self.views['xz'].view.setYLink(self.views['z'])
            self.views['xz'].view.setXLink(self.views['xy'].view)

        if len(self.data.dims) == 4:
            self.generate_marginal_for((1, 3), 0, 0, 'xz')
            self.generate_marginal_for((2, 3), 1, 0, 'xy', cursors=True)
            self.generate_marginal_for((0, 2,), 1, 1, 'yz')
            self.generate_marginal_for((0, 1,), 0, 1, 'zw', cursors=True)

    def set_colormap(self, colormap):
        sampled_colormap = colormap.colors[::51]
        cmap = pg.ColorMap(pos=np.linspace(0, 1, len(sampled_colormap)), color=sampled_colormap)
        for view_name, view in self.views.items():
            if isinstance(view, DataArrayImageView):
                view.setColorMap(cmap)

    def connect_cursor(self, dimension, the_line):
        self.registered_cursors[dimension].append(the_line)

        def connected_cursor(line: CursorRegion):
            new_cursor = list(self.context['cursor'])
            new_cursor[dimension] = line.getRegion()[0]
            self.update_cursor_position(new_cursor)

        the_line.sigRegionChanged.connect(connected_cursor)

    def update_cursor_position(self, new_cursor, force=False):
        old_cursor = list(self.context['cursor'])
        self.context['cursor'] = new_cursor

        def index_to_value(value, i):
            d = self.data.dims[i]
            c = self.data.coords[d].values
            return c[0] + value * (c[1] - c[0])

        self.context['value_cursor'] = [index_to_value(v, i) for i, v in enumerate(new_cursor)]

        changed_dimensions = [i for i, (x, y) in enumerate(zip(old_cursor, new_cursor)) if x != y]

        cursor_text = ','.join('{}: {:.4g}'.format(x, y)
                               for x, y in zip(self.data.dims, self.context['value_cursor']))
        self.window.statusBar().showMessage('({})'.format(cursor_text))

        # update axis info widgets
        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        # update data
        def safe_slice(vlow, vhigh, axis=0):
            vlow, vhigh = int(min(vlow, vhigh)), int(max(vlow, vhigh))
            rng = len(self.data.coords[self.data.dims[axis]])
            vlow, vhigh = clamp(vlow, 0, rng), clamp(vhigh, 0, rng)
            
            if vlow == vhigh:
                vhigh = vlow + 1

            vlow, vhigh = clamp(vlow, 0, rng), clamp(vhigh, 0, rng)

            if vlow == vhigh:
                vlow = vhigh - 1

            return slice(vlow, vhigh)

        for reactive in self.reactive_views:
            if len(set(reactive.dims).intersection(set(changed_dimensions))) or force:
                try:
                    select_coord = dict(zip([self.data.dims[i] for i in reactive.dims],
                                            [safe_slice(int(new_cursor[i]), int(new_cursor[i] + self.binning[i]), i)
                                             for i in reactive.dims]))
                    if isinstance(reactive.view, DataArrayImageView):
                        image_data = self.data.isel(**select_coord)
                        if len(select_coord):
                            image_data = image_data.mean(list(select_coord.keys()))
                        reactive.view.setImage(image_data)

                    elif isinstance(reactive.view, pg.PlotWidget):
                        for_plot = self.data.isel(**select_coord)
                        if len(select_coord):
                            for_plot = for_plot.mean(list(select_coord.keys()))
                        for_plot = for_plot.values

                        cursors = [l for l in reactive.view.getPlotItem().items if isinstance(l, CursorRegion)]
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
                cursor = CursorRegion(
                    orientation=CursorRegion.Vertical if orientation == 'vert' else CursorRegion.Horizontal,
                    movable=True)
                widget.addItem(cursor, ignoreBounds=False)
                self.connect_cursor(remaining_dims[0], cursor)
        else:
            assert(len(remaining_dims) == 2)
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
        self.content_layout.addWidget(widget, column, row)
        return widget

    @property
    def info_tab(self):
        return hlayout()

    def construct_axes_tab(self):
        inner_items = [AxisInfoWidget(axis_index=i, root=self) for i in range(len(self.data.dims))]
        return hlayout(*inner_items), inner_items

    def construct_binning_tab(self):
        binning_options = QtWidgets.QLabel('Options')
        inner_items = [BinningInfoWidget(axis_index=i, root=self) for i in range(len(self.data.dims))]

        return hlayout(binning_options, *inner_items), inner_items

    def add_contextual_widgets(self):
        self.axes_tab, self.axis_info_widgets = self.construct_axes_tab()
        self.binning_tab, self.binning_info_widgets = self.construct_binning_tab()

        self.tabs = tabs(
            ['Info', self.info_tab,],
            ['Axes', self.axes_tab,],
            ['Binning', self.binning_tab,],
        )
        self.tabs.setFixedHeight(150)

        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(self.tabs, 1, 0)

    def start(self):
        app = QtGui.QApplication([])

        win = QtToolWindow()
        win.resize(1100, 1100)
        win.setWindowTitle('QT Tool')
        cw = QtGui.QWidget()
        win.setCentralWidget(cw)
        self.window = win
        self.window.tool = self

        self.content_layout = QtGui.QGridLayout()
        self.main_layout = QtGui.QGridLayout()

        cw.setLayout(self.main_layout)

        # add main image widgets
        self.configure_image_widgets()
        self.add_contextual_widgets()
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
        self._binning = [1 for _ in self.data.dims]


def qt_tool(data: DataType):
    tool = QtTool()
    tool.set_data(data)
    tool.start()


