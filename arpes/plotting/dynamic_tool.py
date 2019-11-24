import inspect

from PyQt5 import QtWidgets

from arpes.utilities import normalize_to_spectrum, group_by
from arpes.typing import DataType

from arpes.utilities.qt import qt_info, SimpleApp, SimpleWindow, BasicHelpDialog
from arpes.utilities.ui import CollectUI, tabs, horizontal, numeric_input, line_edit, label, vertical

__all__ = ('DynamicTool', 'make_dynamic',)

qt_info.setup_pyqtgraph()


class DynamicToolWindow(SimpleWindow):
    HELP_DIALOG_CLS = BasicHelpDialog


class DynamicTool(SimpleApp):
    WINDOW_SIZE = (5, 6.5,) # 5 inches by 5 inches
    WINDOW_CLS = DynamicToolWindow
    TITLE = '' # we will use the function name for the window title

    def __init__(self, function, meta=None):
        self._function = function
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        self.meta = meta or {}
        self.current_arguments = {}

        super().__init__()

    def layout(self):
        return self.main_layout

    def configure_image_widgets(self):
        self.generate_marginal_for((), 0, 0, 'xy', cursors=False, layout=self.content_layout)
        self.generate_marginal_for((), 1, 0, 'f(xy)', cursors=False, layout=self.content_layout)
        self.main_layout.addLayout(self.content_layout, 0, 0)

    def update_data(self):
        self.views['xy'].setImage(self.data.S.nan_to_num())
        try:
            mapped_data = self._function(self.data, **self.current_arguments)
            self.views['f(xy)'].setImage(mapped_data.S.nan_to_num())
        except:
            pass

    def add_controls(self):
        specification = self.calculate_control_specification()

        ui = {}
        with CollectUI(ui):
            controls = tabs(
                ['Controls', horizontal(
                    *[vertical(*[vertical(label(s[0]), self.build_control_for(*s)) for s in pair])
                      for pair in group_by(2, specification)])],
            )

        def update_argument(arg_name, arg_type):
            def updater(value):
                self.current_arguments[arg_name] = arg_type(value)
                self.update_data()

            return updater

        for arg_name, arg_type, _ in specification:
            ui[f'{arg_name}-control'].subject.subscribe(update_argument(arg_name, arg_type))

        controls.setFixedHeight(qt_info.inches_to_px(1.4))
        self.main_layout.addWidget(controls, 1, 0)

    def calculate_control_specification(self):
        argspec = inspect.getfullargspec(self._function)

        # we assume that the first argument is the input data
        args = argspec.args[1:]

        defaults_for_type = {
            float: 0.,
            int: 0,
            str: '',
        }

        specs = []
        for i, arg in enumerate(args[::-1]):
            argument_type = argspec.annotations.get(arg, float)
            if i < len(argspec.defaults):
                argument_default = argspec.defaults[len(argspec.defaults) - (i+1)]
            else:
                argument_default = defaults_for_type.get(argument_type, 0)

            self.current_arguments[arg] = argument_default
            specs.append([
                arg,
                argument_type,
                argument_default,
            ])

        return specs

    def build_control_for(self, parameter_name, parameter_type, parameter_default):
        meta = self.meta.get(parameter_name, {})
        if parameter_type in (int, float,):
            config = {}
            if 'min' in meta:
                config['bottom'] = meta['min']
            if 'max' in meta:
                config['top'] = meta['max']
            return numeric_input(parameter_default, parameter_type, validator_settings=config, id=f'{parameter_name}-control')

        if parameter_type == str:
            return line_edit(parameter_default, id=f'{parameter_name}-control')

    def before_show(self):
        self.configure_image_widgets()
        self.add_controls()
        self.update_data()
        self.window.setWindowTitle(f'Interactive {self._function.__name__}')

    def set_data(self, data: DataType):
        self.data = normalize_to_spectrum(data)


def make_dynamic(fn, data):
    tool = DynamicTool(fn)
    tool.set_data(data)
    tool.start()
