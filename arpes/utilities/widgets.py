import os
from rx.subject import BehaviorSubject, Subject

from PyQt5.QtWidgets import (
    QPushButton, QCheckBox, QComboBox,
    QSpinBox, QTextEdit, QSlider,
    QLineEdit, QRadioButton,
    QWidget, QFileDialog, QHBoxLayout
)

__all__ = (
    'SubjectivePushButton', 'SubjectiveCheckBox',
    'SubjectiveComboBox', 'SubjectiveFileDialog',
    'SubjectiveLineEdit', 'SubjectiveRadioButton',
    'SubjectiveSlider', 'SubjectiveSpinBox',
    'SubjectiveTextEdit',
)


class SubjectiveComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.currentData())
        self.currentIndexChanged.connect(lambda: self.subject.on_next(self.currentText()))


class SubjectiveSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.value())
        self.valueChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        self.setValue(value)


class SubjectiveTextEdit(QTextEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.subject = BehaviorSubject(self.toPlainText())
        self.textChanged.connect(lambda: self.subject.on_next(self.toPlainText()))
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        if self.toPlainText() != value:
            self.setPlainText(value)


class SubjectiveSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.value())
        self.valueChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        self.setValue(value)


class SubjectiveLineEdit(QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.subject = BehaviorSubject(self.text())
        self.textChanged[str].connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        if value != self.text():
            self.setText(value)


class SubjectiveRadioButton(QRadioButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.subject = BehaviorSubject(self.isChecked())
        self.toggled.connect(lambda: self.subject.on_next(self.isChecked()))
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        self.setChecked(value)


class SubjectiveFileDialog(QWidget):
    def __init__(self, *args, single=True, dialog_root=None):
        if dialog_root is None:
            dialog_root = os.getcwd()

        super().__init__(*args)

        self.dialog_root = dialog_root
        self.subject = BehaviorSubject(None)

        layout = QHBoxLayout()
        self.btn = SubjectivePushButton('Open')
        if single:
            self.btn.subject.subscribe(on_next=lambda _: self.get_file())
        else:
            self.btn.subject.subscribe(on_next=lambda _: self.get_files())

        layout.addWidget(self.btn)
        self.setLayout(layout)

    def get_file(self):
        filename = QFileDialog.getOpenFileName(
            self, 'Open File', self.dialog_root)

        self.subject.on_next(filename[0])

    def get_files(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.subject.on_next(filenames)


class SubjectivePushButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.subject = Subject()
        self.clicked.connect(lambda: self.subject.on_next(True))


class SubjectiveCheckBox(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.subject = BehaviorSubject(self.checkState())
        self.stateChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value):
        self.setCheckState(value)

