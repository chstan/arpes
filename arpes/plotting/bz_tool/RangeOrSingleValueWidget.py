"""A control which can provide a range or a single value (i.e. a half open range with ends equal)."""
from functools import partial

from PyQt5 import QtGui, QtWidgets


__all__ = ["RangeOrSingleValueWidget"]


class RangeOrSingleValueWidget(QtWidgets.QGroupBox):
    """A UI control letting you set a single value or a range.

    Used for modeling single cuts or multi-cut scans in the BZ tool.
    """

    def __init__(self, parent=None, root=None, coordinate_name=None, value=None):
        """Configures and initializes inner widgts.

        Inernally, we use a checkbox, spinbox, and slider to model the UI controls here,
        the checkbox determines which UI control is the "active" one.
        """
        super().__init__(title=coordinate_name, parent=parent)

        self.layout = QtGui.QGridLayout(self)

        self.label = QtWidgets.QLabel("Value: ")
        self.spinbox = QtWidgets.QSpinBox()
        self.slider = QtWidgets.QSlider()
        self.checkbox = QtWidgets.QCheckBox("Ranged")

        self.root = root

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spinbox)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.toggle)

        self._prevent_change_events = False

        self.slider.valueChanged.connect(partial(self.value_changed, source=self.slider))
        self.spinbox.valueChanged.connect(partial(self.value_changed, source=self.spinbox))
        self.checkbox.valueChanged.connect(partial(self.mode_changed, source=self.checkbox))

        self.recompute()

    def mode_changed(self, event, source):
        """Unused, currently."""
        pass

    def value_changed(self, event, source):
        """Responds to changes in the internal value."""
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
        """Recompute UI representation from inner values."""
        value = self.spinbox.value()
        self.label.setText("Value: {:.3g}".format(value))

    def value(self):
        """The inner value."""
        return self.spinbox.value()
