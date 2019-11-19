# pylint: disable=import-error

from PyQt5 import QtCore, QtWidgets

from arpes.utilities.ui import PRETTY_KEYS, label, vertical

__all__ = ('BasicHelpDialog',)


class BasicHelpDialog(QtWidgets.QDialog):
    def __init__(self, shortcuts=None):
        super().__init__()

        if shortcuts is None:
            shortcuts = []

        self.layout = QtWidgets.QVBoxLayout()

        keyboard_shortcuts_info = QtWidgets.QGroupBox(title='Keyboard Shortcuts')
        keyboard_shortcuts_layout = QtWidgets.QGridLayout()
        for i, shortcut in enumerate(shortcuts):
            keyboard_shortcuts_layout.addWidget(label(
                ', '.join(PRETTY_KEYS[k] for k in shortcut.chord), wordWrap=True), i, 0)
            keyboard_shortcuts_layout.addWidget(label(shortcut.label), i, 1)

        keyboard_shortcuts_info.setLayout(keyboard_shortcuts_layout)

        aboutInfo = QtWidgets.QGroupBox(title='About')
        aboutLayout = vertical(
            label(
                'QtTool is the work of Conrad Stansbury, with much inspiration '
                'and thanks to the authors of ImageTool. QtTool is distributed '
                'as part of the PyARPES data analysis framework.',
                wordWrap=True
            ),
            label(
                'Complaints and feature requests should be directed to chstan@berkeley.edu.',
                wordWrap=True
            ),
        )

        from arpes.utilities.qt import qt_info # circular dependency
        aboutInfo.setFixedHeight(qt_info.inches_to_px(1))

        self.layout.addWidget(keyboard_shortcuts_info)
        self.layout.addWidget(aboutInfo)
        self.setLayout(self.layout)

        self.setWindowTitle(f'Interactive Utility Help')
        self.setFixedSize(*qt_info.inches_to_px([2, 4]))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_H or event.key() == QtCore.Qt.Key_Escape:
            self._main_window._help_dialog = None # pylint: disable=protected-access
            self.close()
