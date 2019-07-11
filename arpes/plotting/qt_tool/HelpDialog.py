from PyQt5 import QtWidgets, QtCore

from .utils import PRETTY_KEYS, vlayout, label

class HelpDialog(QtWidgets.QDialog):
    def __init__(self, shortcuts=None):
        super().__init__()

        if shortcuts is None:
            shortcuts = []

        self.layout = QtWidgets.QVBoxLayout()

        keyboardShortcutsInfo = QtWidgets.QGroupBox(title='Keyboard Shortcuts')
        keyboardShortcutsLayout = QtWidgets.QGridLayout()
        for i, shortcut in enumerate(shortcuts):
            keyboardShortcutsLayout.addWidget(label(
                ', '.join(PRETTY_KEYS[k] for k in shortcut.chord), wordWrap=True), i, 0)
            keyboardShortcutsLayout.addWidget(label(shortcut.label), i, 1)

        keyboardShortcutsInfo.setLayout(keyboardShortcutsLayout)

        aboutInfo = QtWidgets.QGroupBox(title='About')
        aboutLayout = vlayout(
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