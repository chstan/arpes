from functools import partial

from PyQt5 import QtGui, QtWidgets

__all__ = ('CoordinateOffsetWidget',)


class CoordinateOffsetWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None, root=None, coordinate_name=None, value=None):
        super().__init__(title=coordinate_name, parent=parent)

        self.layout = QtGui.QGridLayout(self)

        self.label = QtWidgets.QLabel('Value: ')
        self.spinbox = QtWidgets.QSpinBox()
        self.slider = QtWidgets.QSlider()
        self.root = root

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spinbox)
        self.layout.addWidget(self.slider)

        self._prevent_change_events = False

        self.slider.valueChanged.connect(partial(self.value_changed, source=self.slider))
        self.spinbox.valueChanged.connect(partial(self.value_changed, source=self.spinbox))

        self.recompute()

    def value_changed(self, event, source):
        if self._prevent_change_events:
            return

        self._prevent_change_events = True
        self.slider.setValue(source.value())
        self.spinbox.setValue(source.value())
        self._prevent_change_events = False
        self.recompute()
        if self.root is not None:
            self.root.update_cut()

    def recompute(self):
        value = self.spinbox.value()
        self.label.setText('Value: {:.3g}'.format(value))

    def value(self):
        return self.spinbox.value()