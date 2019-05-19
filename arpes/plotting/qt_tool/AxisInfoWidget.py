from PyQt5 import QtWidgets, QtGui

__all__ = ('AxisInfoWidget',)

class AxisInfoWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None, root=None, axis_index=None):
        super().__init__(title=str(axis_index), parent=parent)

        self.layout = QtGui.QGridLayout(self)

        self.label = QtWidgets.QLabel('Cursor: ')
        self.transpose_button = QtWidgets.QPushButton('To Front')
        self.transpose_button.clicked.connect(self.on_transpose)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.transpose_button)

        self.axis_index = axis_index
        self.root = root
        self.setLayout(self.layout)
        self.recompute()

    def recompute(self):
        self.setTitle(self.root.data.dims[self.axis_index])
        try:
            cursor_index = self.root.context['cursor'][self.axis_index]
            cursor_value = self.root.context['value_cursor'][self.axis_index]
            self.label.setText('Cursor: {}, {:.3g}'.format(int(cursor_index), cursor_value))
        except KeyError:
            pass

    def on_transpose(self):
        try:
            self.root.transpose_to_front(self.axis_index)
        except Exception:
            pass
