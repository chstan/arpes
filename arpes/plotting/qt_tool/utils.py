# pylint: disable=import-error

from collections import namedtuple

from PyQt5 import QtCore, QtWidgets

__all__ = ('PRETTY_KEYS', 'pretty_key_event', 'KeyBinding', 'layout', 'hlayout', 'vlayout', 'tabs', 'label')

KeyBinding = namedtuple('KeyBinding', ('label', 'chord', 'handler'))

PRETTY_KEYS = {}
for key, value in vars(QtCore.Qt).items():
    if isinstance(value, QtCore.Qt.Key):
        PRETTY_KEYS[value] = key.partition('_')[2]


def pretty_key_event(event):
    """
    Key Event -> List[str] in order to be able to prettily print keys
    :param event:
    :return:
    """
    key_sequence = []

    key_name = PRETTY_KEYS.get(event.key(), event.text())
    if key_name not in key_sequence:
        key_sequence.append(key_name)

    return key_sequence


def tabs(*children):
    widget = QtWidgets.QTabWidget()
    for name, child in children:
        widget.addTab(child, name)

    return widget


def layout(*children, layout_cls=None, widget=None):
    if layout_cls is None:
        layout_cls = QtWidgets.QGridLayout

    if widget is None:
        widget = QtWidgets.QWidget()

    internal_layout = layout_cls()

    for child in children:
        internal_layout.addWidget(child)


    widget.setLayout(internal_layout)

    return widget


def hlayout(*args, **kwargs):
    return layout(*args, **kwargs, layout_cls=QtWidgets.QHBoxLayout)


def vlayout(*args, **kwargs):
    return layout(*args, **kwargs, layout_cls=QtWidgets.QVBoxLayout)


def label(text, **kwargs):
    return QtWidgets.QLabel(text, **kwargs)
