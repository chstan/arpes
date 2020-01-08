"""
Easily composable and reactive UI utilities using RxPy and PyQt5. This makes UI prototyping *MUCH* faster.
In order to log IDs so that you can attach subscriptions after the fact, you will need to use the CollectUI
context manager.

An example is as follows, showing the currently available widgets. If you don't need to attach callbacks,
you can get away without using the context manager.

```
ui = {}
with CollectUI(ui):
    test_widget = grid(
        group(
            text_edit('starting text', id='text'),
            line_edit('starting line', id='line'),
            combo_box(['A', 'B', 'C'], id='combo'),
            spin_box(5, id='spinbox'),
            radio_button('A Radio', id='radio'),
            check_box('Checkbox', id='check'),
            slider(id='slider'),
            file_dialog(id='file'),
            button('Send Text', id='submit')
        ),
        widget=self,
    )
```

"Forms" can effectively be built by building an observable out of the subjects in the UI. We have a `submit`
function that makes creating such an observable simple.

```
submit('submit', ['check', 'slider', 'file'], ui).subscribe(lambda item: print(item))
```

With the line above, whenever the button with id='submit' is pressed, we will log a dictionary with
the most recent values of the inputs {'check','slider','file'} as a dictionary with these keys. This
allows building PyQt5 "forms" without effort.
"""
from enum import Enum
from typing import Type

import enum

import rx.operators as ops
import rx

import pyqtgraph as pg

from typing import Dict, List, Optional
from collections import namedtuple

import functools
from PyQt5.QtWidgets import (
    QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QSplitter,
    QGroupBox, QLabel,
)

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt

from .widgets import *

__all__ = (
    'CollectUI',

    'CursorRegion',

    # layouts
    'layout', 'grid', 'vertical', 'horizontal',

    'splitter',

    # widgets
    'group', 'label', 'tabs', 'button', 'check_box', 'combo_box', 'file_dialog',
    'line_edit', 'radio_button', 'slider', 'spin_box', 'text_edit', 'numeric_input',

    # Observable tools
    'submit',

     # @dataclass utils
    'layout_dataclass',
    'bind_dataclass',

    # Keybinding
    'PRETTY_KEYS',
    'pretty_key_event',
    'KeyBinding',
)


KeyBinding = namedtuple('KeyBinding', ('label', 'chord', 'handler'))
CursorMode = namedtuple('CursorMode', ('label', 'chord', 'handler', 'supported_dimensions'))

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



ACTIVE_UI = None

def ui_builder(f):
    @functools.wraps(f)
    def wrapped_ui_builder(*args, id=None, **kwargs):
        global ACTIVE_UI
        if id is not None:
            try:
                id, ui = id
            except ValueError:
                id, ui = id, ACTIVE_UI

        ui_element = f(*args, **kwargs)

        if id:
            ui[id] = ui_element

        return ui_element

    return wrapped_ui_builder


class CollectUI:
    def __init__(self, target_ui=None):
        global ACTIVE_UI
        assert ACTIVE_UI is None

        self.ui = {} if target_ui is None else target_ui
        ACTIVE_UI = self.ui

    def __enter__(self):
        return self.ui

    def __exit__(self, exc_type, exc_val, exc_tb):
        global ACTIVE_UI
        ACTIVE_UI = None


@ui_builder
def layout(*children, layout_cls=None, widget=None):
    if layout_cls is None:
        layout_cls = QGridLayout

    if widget is None:
        widget = QWidget()

    internal_layout = layout_cls()

    for child in children:
        internal_layout.addWidget(_wrap_text(child))

    widget.setLayout(internal_layout)

    return widget

grid = functools.partial(layout, layout_cls=QGridLayout)
vertical = functools.partial(layout, layout_cls=QVBoxLayout)
horizontal = functools.partial(layout, layout_cls=QHBoxLayout)

@ui_builder
def splitter(first, second, direction=Qt.Vertical, size=None):
    split_widget = QSplitter(direction)

    split_widget.addWidget(first)
    split_widget.addWidget(second)

    if size is not None:
        split_widget.setSizes(size)

    return split_widget

splitter.Vertical = Qt.Vertical
splitter.Horizontal = Qt.Horizontal


@ui_builder
def group(*args, label=None, layout_cls=None):
    if args:
        if isinstance(args[0], str):
            label = args[0]
            args = args[1:]

    if layout_cls is None:
        layout_cls = QVBoxLayout

    groupbox = QGroupBox(label)

    layout = layout_cls()

    for arg in args:
        layout.addWidget(arg)

    groupbox.setLayout(layout)
    return groupbox

@ui_builder
def label(text, *args, **kwargs):
    return QLabel(text, *args, **kwargs)

@ui_builder
def tabs(*children):
    widget = QTabWidget()
    for name, child in children:
        widget.addTab(child, name)

    return widget

@ui_builder
def button(text, *args):
    return SubjectivePushButton(text, *args)

@ui_builder
def check_box(text, *args):
    return SubjectiveCheckBox(text, *args)

@ui_builder
def combo_box(items, *args, name=None):
    widget = SubjectiveComboBox(*args)
    widget.addItems(items)

    if name is not None:
        widget.setObjectName(name)

    return widget

@ui_builder
def file_dialog(*args):
    return SubjectiveFileDialog(*args)

@ui_builder
def line_edit(*args):
    return SubjectiveLineEdit(*args)

@ui_builder
def radio_button(text, *args):
    return SubjectiveRadioButton(text, *args)

@ui_builder
def slider(minimum=0, maximum=10, interval=None, horizontal=True):
    widget = SubjectiveSlider(orientation=Qt.Horizontal if horizontal else Qt.Vertical)
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)

    if interval:
        widget.setTickInterval(interval)

    return widget

@ui_builder
def spin_box(minimum=0, maximum=10, step=1, adaptive=True, value=None):
    widget = SubjectiveSpinBox()

    widget.setRange(minimum, maximum)

    if value is not None:
        widget.subject.on_next(value)

    if adaptive:
        widget.setStepType(SubjectiveSpinBox.AdaptiveDecimalStepType)
    else:
        widget.setSingleStep(step)

    return widget

@ui_builder
def text_edit(text='', *args):
    return SubjectiveTextEdit(text, *args)

@ui_builder
def numeric_input(value=0, input_type: type = float, *args, validator_settings=None):
    validators = {
        int: QtGui.QIntValidator,
        float: QtGui.QDoubleValidator,
    }
    default_settings = {
        int: {
            'bottom': -1e6,
            'top': 1e6,
        },
        float: {
            'bottom': -1e6,
            'top': 1e6,
            'decimals': 3,
        }
    }

    if validator_settings is None:
        validator_settings = default_settings.get(input_type)

    widget = SubjectiveLineEdit(str(value), *args)
    widget.setValidator(validators.get(input_type, QtGui.QIntValidator)(**validator_settings))

    return widget

def _wrap_text(str_or_widget):
    return label(str_or_widget) if isinstance(str_or_widget, str) else str_or_widget

def _unwrap_subject(subject_or_widget):
    try:
        return subject_or_widget.subject
    except AttributeError:
        return subject_or_widget

def submit(gate: str, keys: List[str], ui: Dict[str, QWidget]) -> rx.Observable:
    if isinstance(gate, str):
        gate = ui[gate]

    gate = _unwrap_subject(gate)
    items = [_unwrap_subject(ui[k]) for k in keys]

    combined = items[0].pipe(
        ops.combine_latest(*items[1:]),
        ops.map(lambda vs: dict(zip(keys, vs)))
    )

    return gate.pipe(
        ops.filter(lambda x: x),
        ops.with_latest_from(combined),
        ops.map(lambda x: x[1])
    )

def _try_unwrap_value(v):
    try:
        return v.value
    except AttributeError:
        return v


def enum_option_names(enum_cls: Type[enum.Enum]) -> List[str]:
    names = [x for x in dir(enum_cls) if '__' not in x]
    values = [_try_unwrap_value(getattr(enum_cls, n)) for n in names]

    return [x[0] for x in sorted(zip(names, values), key=lambda x: x[1])]


def enum_mapping(enum_cls: Type[enum.Enum], invert=False):
    options = enum_option_names(enum_cls)
    d = dict([[o, _try_unwrap_value(getattr(enum_cls, o))] for o in options])
    if invert:
        d = {v: k for k, v in d.items()}
    return d


def _layout_dataclass_field(dataclass_cls, field_name: str, prefix: str):
        id_for_field = f'{prefix}.{field_name}'

        field = dataclass_cls.__dataclass_fields__[field_name]
        if field.type in [int, float,]:
            field_input = numeric_input(value=0, input_type=field.type, id=id_for_field)
        elif field.type == str:
            field_input = line_edit('', id=id_for_field)
        elif issubclass(field.type, enum.Enum):
            enum_options = enum_option_names(field.type)
            field_input = combo_box(enum_options, id=id_for_field)
        elif field.type == bool:
            field_input = check_box(field_name, id=id_for_field)
        else:
            raise Exception('Could not render field: {}'.format(field))

        return group(
            field_name,
            field_input,
        )

def layout_dataclass(dataclass_cls, prefix: Optional[str] = None):
    """
    Renders a dataclass instance to QtWidgets. See also `bind_dataclass` below
    to get one way data binding to the instance
    :param dataclass_cls:
    :param prefix:
    :return:
    """
    if prefix is None:
        prefix = dataclass_cls.__name__

    return vertical(
        *[_layout_dataclass_field(dataclass_cls, field_name, prefix)
          for field_name in dataclass_cls.__dataclass_fields__]
    )

def bind_dataclass(dataclass_instance, prefix: str, ui: Dict[str, QWidget]):
    """
    One-way data binding between a dataclass instance and a collection of widgets in the UI.

    Sets the current UI state to the value of the Python dataclass instance, and sets up
    subscriptions to value changes on the UI so that any future changes are propagated to
    the dataclass instance.

    :param dataclass_instance: Instance to link
    :param prefix: Prefix for widget IDs in the UI
    :param ui: Collected UI elements
    :return:
    """
    relevant_widgets = {k.split(prefix)[1]: v  for k, v in ui.items() if k.startswith(prefix)}
    for field_name, field in dataclass_instance.__dataclass_fields__.items():
        translate_from_field, translate_to_field = {
            int: (lambda x: str(x), lambda x: int(x)),
            float: (lambda x: str(x), lambda x: float(x)),
        }.get(field.type, (lambda x: x, lambda x: x))

        if issubclass(field.type, Enum):
            forward_mapping = dict(sorted(enum_mapping(field.type).items(), key=lambda x: int(x[1])))
            inverse_mapping = {v: k for k, v in forward_mapping.items()}

            def extract_field(v):
                try:
                    return v.value
                except AttributeError:
                    return v

            translate_to_field = lambda x: forward_mapping[x]
            translate_from_field = lambda x: inverse_mapping[extract_field(x)]

        current_value = translate_from_field(getattr(dataclass_instance, field_name))
        w = relevant_widgets[field_name]

        # write the current value to the UI
        w.subject.on_next(current_value)

        # close over the translation function
        def build_setter(translate, name):
            def setter(value):
                try:
                    value = translate(value)
                except ValueError:
                    return

                setattr(dataclass_instance, name, value)

            return setter

        w.subject.subscribe(build_setter(translate_to_field, field_name))


class CursorRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._region_width = 1
        self.lines[1].setMovable(False)

    def set_width(self, value):
        self._region_width = value
        self.lineMoved(0)

    def lineMoved(self, i):
        if self.blockLineSignal:
            return

        self.lines[1].setValue(self.lines[0].value() + self._region_width)
        self.prepareGeometryChange()
        self.sigRegionChanged.emit(self)

    def set_location(self, value):
        old = self.blockLineSignal
        self.blockLineSignal = True
        self.lines[1].setValue(value + self._region_width)
        self.lines[0].setValue(value + self._region_width)
        self.blockLineSignal = old
