"""Wraps Qt widgets in ones which use rx for signaling, Conrad's personal preference."""
import os
from rx.subject import BehaviorSubject, Subject

from PyQt5.QtWidgets import (
    QPushButton,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QTextEdit,
    QSlider,
    QLineEdit,
    QRadioButton,
    QWidget,
    QFileDialog,
    QHBoxLayout,
)

__all__ = (
    "SubjectivePushButton",
    "SubjectiveCheckBox",
    "SubjectiveComboBox",
    "SubjectiveFileDialog",
    "SubjectiveLineEdit",
    "SubjectiveRadioButton",
    "SubjectiveSlider",
    "SubjectiveSpinBox",
    "SubjectiveTextEdit",
)


class SubjectiveComboBox(QComboBox):
    """A QComboBox using rx instead of signals."""

    def __init__(self, *args, **kwargs):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.currentData())
        self.currentIndexChanged.connect(lambda: self.subject.on_next(self.currentText()))


class SubjectiveSpinBox(QSpinBox):
    """A QSpinBox using rx instead of signals."""

    def __init__(self, *args, **kwargs):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.value())
        self.valueChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        """Forwards value change to the UI."""
        self.setValue(value)


class SubjectiveTextEdit(QTextEdit):
    """A QTextEdit using rx instead of signals."""

    def __init__(self, *args):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.toPlainText())
        self.textChanged.connect(lambda: self.subject.on_next(self.toPlainText()))
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        """Forwards value change to the UI."""
        if self.toPlainText() != value:
            self.setPlainText(value)


class SubjectiveSlider(QSlider):
    """A QSlider using rx instead of signals."""

    def __init__(self, *args, **kwargs):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.value())
        self.valueChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        """Forwards value change to the UI."""
        self.setValue(value)


class SubjectiveLineEdit(QLineEdit):
    """A QLineEdit using rx instead of signals."""

    def __init__(self, *args):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.text())
        self.textChanged[str].connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        """Forwards value change to the UI."""
        if value != self.text():
            self.setText(value)


class SubjectiveRadioButton(QRadioButton):
    """A QRadioButton using rx instead of signals."""

    def __init__(self, *args):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.isChecked())
        self.toggled.connect(lambda: self.subject.on_next(self.isChecked()))
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        """Forwards value change to the UI."""
        self.setChecked(value)


class SubjectiveFileDialog(QWidget):
    """A file dialog implemented in Qt supporting single or multiple file selection."""

    def __init__(self, *args, single=True, dialog_root=None):
        """Sets up the file dialog widget.

        In addition to wrapping signals in BehaviorSubject as we do elsewhere,
        we need to get a reasonable place to open dialogs when they are requested.

        This can be configured with `dialog_root`, otherwise we will use `os.getcwd()`.
        """
        if dialog_root is None:
            dialog_root = os.getcwd()

        super().__init__(*args)

        self.dialog_root = dialog_root
        self.subject = BehaviorSubject(None)

        layout = QHBoxLayout()
        self.btn = SubjectivePushButton("Open")
        if single:
            self.btn.subject.subscribe(on_next=lambda _: self.get_file())
        else:
            self.btn.subject.subscribe(on_next=lambda _: self.get_files())

        layout.addWidget(self.btn)
        self.setLayout(layout)

    def get_file(self):
        """Opens a dialog allowing a single from the user."""
        filename = QFileDialog.getOpenFileName(self, "Open File", self.dialog_root)

        self.subject.on_next(filename[0])

    def get_files(self):
        """Opens a dialog allowing multiple selections from the user."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.subject.on_next(filenames)


class SubjectivePushButton(QPushButton):
    """A QCheckBox using rx instead of signals."""

    def __init__(self, *args, **kwargs):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = Subject()
        self.clicked.connect(lambda: self.subject.on_next(True))


class SubjectiveCheckBox(QCheckBox):
    """A QCheckBox using rx instead of signals."""

    def __init__(self, *args, **kwargs):
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.checkState())
        self.stateChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        """Forwards value change to the UI."""
        self.setCheckState(value)
