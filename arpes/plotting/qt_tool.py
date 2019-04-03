from PyQt5 import QtGui, QtCore, QtWidgets
import PyQt5.QtGui
import pyqtgraph as pg
import numpy as np
from collections import namedtuple, defaultdict

from arpes.utilities import normalize_to_spectrum
from arpes.typing import DataType
import arpes.config

__all__ = ('QtTool', 'qt_tool',)

ReactivePlotRecord = namedtuple('ReactivePlotRecord', ('dims', 'view', 'orientation',))

pg.setConfigOptions(antialias=True, foreground=(0, 0, 0), background=(255, 255, 255))

pretty_keys = {}
for key, value in vars(QtCore.Qt).items():
    if isinstance(value, QtCore.Qt.Key):
        pretty_keys[value] = key.partition('_')[2]

def pretty_key_event(event):
    """
    Key Event -> List[str] in order to be able to prettily print keys
    :param event:
    :return:
    """
    key_sequence = []

    key = pretty_keys.get(event.key(), event.text())
    if key not in key_sequence:
        key_sequence.append(key)

    return key_sequence

#class AxisInfoWidget

class HelpDialog(QtWidgets.QDialog):
    def __init__(self, shortcuts=None):
        super().__init__()

        if shortcuts is None:
            shortcuts = []

        self.layout = QtWidgets.QVBoxLayout()

        keyboardShortcutsInfo = QtWidgets.QGroupBox(title='Keyboard Shortcuts')
        keyboardShortcutsLayout = QtWidgets.QGridLayout()
        for i, shortcut in enumerate(shortcuts):

            keyboardShortcutsLayout.addWidget(QtWidgets.QLabel(
                ', '.join(pretty_keys[k] for k in shortcut.chord), wordWrap=True), i, 0)
            keyboardShortcutsLayout.addWidget(QtWidgets.QLabel(shortcut.label), i, 1)

        keyboardShortcutsInfo.setLayout(keyboardShortcutsLayout)

        aboutInfo = QtWidgets.QGroupBox(title='About')
        aboutLayout = QtWidgets.QVBoxLayout()
        aboutLayout.addWidget(QtWidgets.QLabel(
            'QtTool is the work of Conrad Stansbury, with much inspiration '
            'and thanks to the authors of ImageTool. QtTool is distributed '
            'as part of the PyPES data analysis framework.',
            wordWrap=True
        ))
        aboutLayout.addWidget(QtWidgets.QLabel(
            'Complaints and feature requests should be directed to chstan@berkeley.edu.',
            wordWrap=True
        ))
        aboutInfo.setLayout(aboutLayout)
        aboutInfo.setFixedHeight(150)

        self.layout.addWidget(keyboardShortcutsInfo)
        self.layout.addWidget(aboutInfo)
        self.setLayout(self.layout)

        self.setWindowTitle('QtTool - Help')
        self.setFixedSize(300, 500)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_H or event.key() == QtCore.Qt.Key_Escape:
            self._main_window._help_dialog = None
            self.close()

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


class DataArrayImageView(pg.ImageView):
    """
    ImageView that transparently handles xarray data, including setting axis and coordinate information.

    This makes it easier to build interactive applications around realistic scientific datasets.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(view=pg.PlotItem(), *args, **kwargs)

        self.view.invertY(False)

    def setImage(self, img, *args, **kwargs):
        """
        Accepts an xarray.DataArray instead of a numpy array
        :param img:
        :param args:
        :param kwargs:
        :return:
        """

        super().setImage(img.values, *args, **kwargs)
        #self.view.axis


KeyBinding = namedtuple('KeyBinding', ('label', 'chord', 'handler'))


class QtToolWindow(QtGui.QMainWindow, QtCore.QObject):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._help_dialog = None

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

    def scroll(self, event):
        print('Scrolling.')

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

        self.settings = arpes.config.SETTINGS.copy()

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
                    if isinstance(reactive.view, DataArrayImageView):
                        reactive.view.setImage(self.data.isel(**select_coord))
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
            widget = DataArrayImageView(name=name)
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
        self.content_layout.addWidget(widget, column, row)
        return widget

    @property
    def info_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(QtWidgets.QWidget())
        layout.addWidget(QtWidgets.QWidget())
        layout.addWidget(QtWidgets.QWidget())

        return tab

    @property
    def axes_tab(self):
        return QtWidgets.QWidget()

    def add_contextual_widgets(self):
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setFixedHeight(150)

        self.tabs.addTab(self.info_tab, 'Info')
        self.tabs.addTab(self.axes_tab, 'Axes')

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


def qt_tool(data: DataType):
    tool = QtTool()
    tool.set_data(data)
    tool.start()