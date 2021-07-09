"""A widget providing rudimentary information about an axis on a DataArray."""
# pylint: disable=import-error

from PyQt5 import QtGui, QtWidgets

__all__ = ("AxisInfoWidget",)


class AxisInfoWidget(QtWidgets.QGroupBox):
    """A widget providing some rudimentary axis information."""

    def __init__(self, parent=None, root=None, axis_index=None):
        """Configure inner widgets for axis info, and transpose to front button."""
        super().__init__(title=str(axis_index), parent=parent)

        self.layout = QtGui.QGridLayout(self)

        self.label = QtWidgets.QLabel("Cursor: ")
        self.transpose_button = QtWidgets.QPushButton("To Front")
        self.transpose_button.clicked.connect(self.on_transpose)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.transpose_button)

        self.axis_index = axis_index
        self._root = root
        self.setLayout(self.layout)
        self.recompute()

    @property
    def root(self):
        """Unwraps the weakref to the parent application."""
        return self._root()

    def recompute(self):
        """Force a recomputation of dependent UI state: here, the title and text."""
        self.setTitle(self.root.data.dims[self.axis_index])
        try:
            cursor_index = self.root.context["cursor"][self.axis_index]
            cursor_value = self.root.context["value_cursor"][self.axis_index]
            self.label.setText("Cursor: {}, {:.3g}".format(int(cursor_index), cursor_value))
        except KeyError:
            pass

    def on_transpose(self):
        """This UI control lets you tranpose the axis it refers to to the front."""
        try:
            self.root.transpose_to_front(self.axis_index)
        except Exception:
            pass
