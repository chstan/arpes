"""Provides a Qt based implementation of Igor's ImageTool."""
# pylint: disable=import-error

from arpes.utilities.qt.utils import PlotOrientation
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import weakref
import warnings
import dill
from typing import List, Union

import arpes.config
import arpes.xarray_extensions
from arpes.utilities import normalize_to_spectrum
from arpes.typing import DataType
from arpes.utilities.qt.data_array_image_view import DataArrayPlot

from arpes.utilities.ui import KeyBinding, horizontal, tabs, CursorRegion
from arpes.utilities.qt import (
    qt_info,
    run_tool_in_daemon_process,
    DataArrayImageView,
    BasicHelpDialog,
    SimpleWindow,
    SimpleApp,
)

from .AxisInfoWidget import AxisInfoWidget
from .BinningInfoWidget import BinningInfoWidget

__all__ = (
    "QtTool",
    "qt_tool",
)

qt_info.setup_pyqtgraph()


class QtToolWindow(SimpleWindow):
    """The application window for `QtTool`.

    QtToolWindow was the first Qt-Based Tool that I built for PyARPES. Much of its structure was ported
    to SimpleWindow and borrowed ideas from when I wrote DAQuiri. As a result, the structure is essentially
    now to define just the handlers and any lifecycle hooks (close, etc.)
    """

    HELP_DIALOG_CLS = BasicHelpDialog

    def compile_key_bindings(self):
        return super().compile_key_bindings() + [  # already includes Help and Close
            KeyBinding(
                "Scroll Cursor",
                [
                    QtCore.Qt.Key_Left,
                    QtCore.Qt.Key_Right,
                    QtCore.Qt.Key_Up,
                    QtCore.Qt.Key_Down,
                ],
                self.scroll,
            ),
            KeyBinding(
                "Reset Intensity",
                [QtCore.Qt.Key_I],
                self.reset_intensity,
            ),
            KeyBinding(
                "Scroll Z-Cursor",
                [
                    QtCore.Qt.Key_N,
                    QtCore.Qt.Key_M,
                ],
                self.scroll_z,
            ),
            KeyBinding(
                "Center Cursor",
                [QtCore.Qt.Key_C],
                self.center_cursor,
            ),
            KeyBinding(
                "Transpose - Roll Axis",
                [QtCore.Qt.Key_T],
                self.transpose_roll,
            ),
            KeyBinding(
                "Transpose - Swap Front Axes",
                [QtCore.Qt.Key_Y],
                self.transpose_swap,
            ),
        ]

    def center_cursor(self, event):
        self.app().center_cursor()

    def transpose_roll(self, event):
        self.app().transpose_to_front(-1)

    def transpose_swap(self, event):
        self.app().transpose_to_front(1)

    @staticmethod
    def _update_scroll_delta(delta, event: QtGui.QKeyEvent):
        if event.nativeModifiers() & 1:  # shift key
            delta = (delta[0], delta[1] * 5)

        if event.nativeModifiers() & 2:  # shift key
            delta = (delta[0], delta[1] * 20)

        return delta

    def reset_intensity(self, event: QtGui.QKeyEvent):
        self.app().reset_intensity()

    def scroll_z(self, event: QtGui.QKeyEvent):
        key_map = {
            QtCore.Qt.Key_N: (2, -1),
            QtCore.Qt.Key_M: (2, 1),
        }

        delta = self._update_scroll_delta(key_map.get(event.key()), event)

        if delta is not None and self.app() is not None:
            self.app().scroll(delta)

    def scroll(self, event: QtGui.QKeyEvent):
        key_map = {
            QtCore.Qt.Key_Left: (0, -1),
            QtCore.Qt.Key_Right: (0, 1),
            QtCore.Qt.Key_Down: (1, -1),
            QtCore.Qt.Key_Up: (1, 1),
        }

        delta = self._update_scroll_delta(key_map.get(event.key()), event)

        if delta is not None and self.app() is not None:
            self.app().scroll(delta)


class QtTool(SimpleApp):
    """QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PyQt5.

    For now we retain a number of the metaphors from BokehTool, including a "context"
    that stores the state, and can be used to programmatically interface with the tool.
    """

    TITLE = "Qt Tool"
    WINDOW_CLS = QtToolWindow
    WINDOW_SIZE = (5, 5)

    def __init__(self):
        """Initialize attributes to safe empty values."""
        super().__init__()
        self.data = None

        self.content_layout = None
        self.main_layout = None

        self.axis_info_widgets = []
        self.binning_info_widgets = []
        self.kspace_info_widgets = []

        self._binning = None

    def center_cursor(self):
        """Scrolls so that the cursors are in the center of the data volume."""
        new_cursor = [len(self.data.coords[d]) / 2 for d in self.data.dims]
        self.update_cursor_position(new_cursor)

        for i, cursors in self.registered_cursors.items():
            for cursor in cursors:
                cursor.set_location(new_cursor[i])

    def scroll(self, delta):
        """Scroll the axis delta[0] by delta[1] pixels."""
        if delta[0] >= len(self.context["cursor"]):
            warnings.warn("Tried to scroll a non-existent dimension.")
            return

        cursor = list(self.context["cursor"])
        cursor[delta[0]] += delta[1]

        self.update_cursor_position(cursor)

        for i, cursors in self.registered_cursors.items():
            for c in cursors:
                c.set_location(cursor[i])

    @property
    def binning(self):
        """The binning on each axis in pixels."""
        if self._binning is None:
            return [1 for _ in self.data.dims]

        return list(self._binning)

    @binning.setter
    def binning(self, value):
        """Set the desired axis binning."""
        different_binnings = [i for i, (nv, v) in enumerate(zip(value, self._binning)) if nv != v]
        self._binning = value

        for i in different_binnings:
            cursors = self.registered_cursors.get(i)
            for cursor in cursors:
                cursor.set_width(self._binning[i])

        self.update_cursor_position(self.context["cursor"], force=True)

    def transpose(self, transpose_order: List[str]):
        """Transpose dimensions into the order specified by `transpose_order` and redraw."""
        reindex_order = [self.data.dims.index(t) for t in transpose_order]
        self.data = self.data.transpose(*transpose_order)

        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        new_cursor = [self.context["cursor"][i] for i in reindex_order]
        self.update_cursor_position(new_cursor, force=True)

        for i, cursors in self.registered_cursors.items():
            for cursor in cursors:
                cursor.set_location(new_cursor[i])

    def transpose_to_front(self, dim: Union[str, int]):
        """Transpose the dimension `dim` to the front so that it is in the main marginal."""
        if not isinstance(dim, str):
            dim = self.data.dims[dim]

        order = list(self.data.dims)
        order.remove(dim)
        order = [dim] + order
        self.transpose(order)

    def configure_image_widgets(self):
        """Configure array marginals for the input data.

        Depending on the array dimensionality, we need a different number and variety
        of marginals. This is as easy as specifying which marginals we select over and
        handling the rest dynamically.

        An additional complexity is that we also handle the cursor registration here.
        """
        if len(self.data.dims) == 2:
            self.generate_marginal_for((), 1, 0, "xy", cursors=True, layout=self.content_layout)
            self.generate_marginal_for(
                (1,), 0, 0, "x", orientation=PlotOrientation.Horizontal, layout=self.content_layout
            )
            self.generate_marginal_for(
                (0,), 1, 1, "y", orientation=PlotOrientation.Vertical, layout=self.content_layout
            )

            self.views["xy"].view.setYLink(self.views["y"])
            self.views["xy"].view.setXLink(self.views["x"])

        if len(self.data.dims) == 3:
            self.generate_marginal_for(
                (1, 2),
                0,
                0,
                "x",
                orientation=PlotOrientation.Horizontal,
                layout=self.content_layout,
            )
            self.generate_marginal_for((1,), 1, 0, "xz", layout=self.content_layout)
            self.generate_marginal_for((2,), 2, 0, "xy", cursors=True, layout=self.content_layout)
            self.generate_marginal_for(
                (0, 1),
                0,
                1,
                "z",
                orientation=PlotOrientation.Horizontal,
                cursors=True,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (0, 2), 2, 2, "y", orientation=PlotOrientation.Vertical, layout=self.content_layout
            )
            self.generate_marginal_for((0,), 2, 1, "yz", layout=self.content_layout)

            self.views["xy"].view.setYLink(self.views["y"])
            self.views["xy"].view.setXLink(self.views["x"])
            self.views["xz"].view.setYLink(self.views["z"])
            self.views["xz"].view.setXLink(self.views["xy"].view)

        if len(self.data.dims) == 4:
            self.generate_marginal_for((1, 3), 0, 0, "xz", layout=self.content_layout)
            self.generate_marginal_for((2, 3), 1, 0, "xy", cursors=True, layout=self.content_layout)
            self.generate_marginal_for((0, 2), 1, 1, "yz", layout=self.content_layout)
            self.generate_marginal_for((0, 1), 0, 1, "zw", cursors=True, layout=self.content_layout)

    def connect_cursor(self, dimension, the_line):
        """Connect a cursor to a line control.

        without weak references we get a circular dependency here
        because `the_line` is owned by a child of `self` but we are
        providing self to a closure which is retained by `the_line`.
        """
        self.registered_cursors[dimension].append(the_line)
        owner = weakref.ref(self)

        def connected_cursor(line: CursorRegion):
            new_cursor = list(owner().context["cursor"])
            new_cursor[dimension] = line.getRegion()[0]
            owner().update_cursor_position(new_cursor)

        the_line.sigRegionChanged.connect(connected_cursor)

    def update_cursor_position(self, new_cursor, force=False, keep_levels=True):
        """Sets the current cursor position.

        Because setting the cursor position changes the marginal data, this is also
        where redrawing originates.

        The way we do this is basically to step through views, recompute the slice for that view
        and set the image/array on the slice.
        """
        old_cursor = list(self.context["cursor"])
        self.context["cursor"] = new_cursor

        def index_to_value(value, i):
            d = self.data.dims[i]
            c = self.data.coords[d].values
            return c[0] + value * (c[1] - c[0])

        self.context["value_cursor"] = [index_to_value(v, i) for i, v in enumerate(new_cursor)]

        changed_dimensions = [i for i, (x, y) in enumerate(zip(old_cursor, new_cursor)) if x != y]

        cursor_text = ",".join(
            "{}: {:.4g}".format(x, y) for x, y in zip(self.data.dims, self.context["value_cursor"])
        )
        self.window.statusBar().showMessage("({})".format(cursor_text))

        # update axis info widgets
        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        # update data
        def safe_slice(vlow, vhigh, axis=0):
            vlow, vhigh = int(min(vlow, vhigh)), int(max(vlow, vhigh))
            rng = len(self.data.coords[self.data.dims[axis]])
            vlow, vhigh = np.clip(vlow, 0, rng), np.clip(vhigh, 0, rng)

            if vlow == vhigh:
                vhigh = vlow + 1

            vlow, vhigh = np.clip(vlow, 0, rng), np.clip(vhigh, 0, rng)

            if vlow == vhigh:
                vlow = vhigh - 1

            return slice(vlow, vhigh)

        for reactive in self.reactive_views:
            if set(reactive.dims).intersection(set(changed_dimensions)) or force:
                try:
                    select_coord = dict(
                        zip(
                            [self.data.dims[i] for i in reactive.dims],
                            [
                                safe_slice(
                                    int(new_cursor[i]), int(new_cursor[i] + self.binning[i]), i
                                )
                                for i in reactive.dims
                            ],
                        )
                    )
                    if isinstance(reactive.view, DataArrayImageView):
                        image_data = self.data.isel(**select_coord)
                        if select_coord:
                            image_data = image_data.mean(list(select_coord.keys()))
                        reactive.view.setImage(image_data, keep_levels=keep_levels)

                    elif isinstance(reactive.view, pg.PlotWidget):
                        for_plot = self.data.isel(**select_coord)
                        if select_coord:
                            for_plot = for_plot.mean(list(select_coord.keys()))

                        cursors = [
                            l
                            for l in reactive.view.getPlotItem().items
                            if isinstance(l, CursorRegion)
                        ]
                        reactive.view.clear()
                        for c in cursors:
                            reactive.view.addItem(c)

                        if isinstance(reactive.view, DataArrayPlot):
                            reactive.view.plot(for_plot)
                            continue

                        if reactive.orientation == PlotOrientation.Horizontal:
                            reactive.view.plot(for_plot.values)
                        else:
                            reactive.view.plot(for_plot.values, range(len(for_plot.values)))
                except IndexError:
                    pass

    def construct_axes_tab(self):
        """Controls for axis order and transposition."""
        inner_items = [
            AxisInfoWidget(axis_index=i, root=weakref.ref(self)) for i in range(len(self.data.dims))
        ]
        return horizontal(*inner_items), inner_items

    def construct_binning_tab(self):
        """This tab controls the degree of binning around the cursor."""
        binning_options = QtWidgets.QLabel("Options")
        inner_items = [
            BinningInfoWidget(axis_index=i, root=weakref.ref(self))
            for i in range(len(self.data.dims))
        ]

        return horizontal(binning_options, *inner_items), inner_items

    def construct_kspace_tab(self):
        """The momentum exploration tab."""
        inner_items = []
        return horizontal(*inner_items), inner_items

    def add_contextual_widgets(self):
        """Adds the widgets for the contextual controls at the bottom."""
        axes_tab, self.axis_info_widgets = self.construct_axes_tab()
        binning_tab, self.binning_info_widgets = self.construct_binning_tab()
        kspace_tab, self.kspace_info_widgets = self.construct_kspace_tab()

        self.tabs = tabs(
            ["Info", horizontal()],
            ["Axes", axes_tab],
            ["Binning", binning_tab],
            ["K-Space", kspace_tab],
        )
        self.tabs.setFixedHeight(qt_info.inches_to_px(1))

        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(self.tabs, 1, 0)

    def layout(self):
        """Initialize the layout components."""
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        return self.main_layout

    def before_show(self):
        """Lifecycle hook for configuration before app show."""
        self.configure_image_widgets()
        self.add_contextual_widgets()
        import matplotlib.cm

        self.set_colormap(matplotlib.cm.viridis)

    def after_show(self):
        """Initialize application state after app show.

        To do this, we need to set the initial cursor location, and call update
        which forces a rerender.
        """
        # basic state initialization
        self.context.update(
            {
                "cursor": [self.data.coords[d].mean().item() for d in self.data.dims],
            }
        )

        # Display the data
        self.update_cursor_position(self.context["cursor"], force=True, keep_levels=False)
        self.center_cursor()

    def reset_intensity(self):
        """Autoscales intensity in each marginal plot."""
        self.update_cursor_position(self.context["cursor"], force=True, keep_levels=False)

    def set_data(self, data: DataType):
        """Sets the current data to a new value and resets binning."""
        data = normalize_to_spectrum(data)

        if np.any(np.isnan(data)):
            warnings.warn("Nan values encountered, copying data and assigning zeros.")
            data = data.fillna(0)

        self.data = data
        self._binning = [1 for _ in self.data.dims]


def _qt_tool(data: DataType, **kwargs):
    """Starts the qt_tool using an input spectrum."""
    try:
        data = dill.loads(data)
    except TypeError:
        pass

    tool = QtTool()
    tool.set_data(data)
    tool.start(**kwargs)


qt_tool = run_tool_in_daemon_process(_qt_tool)
