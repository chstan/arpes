from PyQt5 import QtWidgets, QtGui

from .utils import layout

__all__ = ('BinningInfoWidget',)

class BinningInfoWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None, root=None, axis_index=None):
        super().__init__(title=str(axis_index), parent=parent)
        self.root = root
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

    def recompute(self):
        self.setTitle(self.root.data.dims[self.axis_index])

    def changeBinning(self):
        try:
            old_binning = self.root.binning
            old_binning[self.axis_index] = self.spinbox.value()
            self.root.binning = old_binning
        except:
            pass