"""An axis binning control."""

from PyQt5 import QtWidgets

from arpes.utilities.ui import layout

__all__ = ("BinningInfoWidget",)


class BinningInfoWidget(QtWidgets.QGroupBox):
    """A spinbox allowing you to set the binning on different axes."""

    def __init__(self, parent=None, root=None, axis_index=None):
        """Initialize an inner spinbox and connect signals to get reactivity."""
        super().__init__(title=str(axis_index), parent=parent)
        self._root = root
        self.axis_index = axis_index

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setMaximum(2000)
        self.spinbox.setMinimum(1)
        self.spinbox.setValue(1)
        self.spinbox.valueChanged.connect(self.changeBinning)
        self.spinbox.editingFinished.connect(self.changeBinning)

        self.layout = layout(
            self.spinbox,
            widget=self,
        )

        self.recompute()

    @property
    def root(self):
        """Unwraps the weakref to the parent application."""
        return self._root()

    def recompute(self):
        """Redraws all dependent UI state, namely the title."""
        self.setTitle(self.root.data.dims[self.axis_index])

    def changeBinning(self):
        """Callback for widget value changes which sets the binning on the root app."""
        try:
            old_binning = self.root.binning
            old_binning[self.axis_index] = self.spinbox.value()
            self.root.binning = old_binning
        except:
            pass
