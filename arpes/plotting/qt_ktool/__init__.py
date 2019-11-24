from PyQt5 import QtWidgets
import numpy as np

from arpes.utilities import normalize_to_spectrum, group_by
from arpes.typing import DataType
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.qt import qt_info, SimpleApp, SimpleWindow
from arpes.utilities.ui import tabs, horizontal, vertical, label, numeric_input, CollectUI

__all__ = ('KTool', 'ktool',)


qt_info.setup_pyqtgraph()


class KTool(SimpleApp):
    """
    QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PyQt5 for now we retain a number of the
    metaphors from BokehTool, including a "context" that stores the state, and can be used to programmatically interface
    with the tool
    """
    TITLE = 'KSpace-Tool'
    WINDOW_SIZE = (5,6,)
    WINDOW_CLS = SimpleWindow

    def __init__(self):
        super().__init__()

        self.conversion_kwargs = {}
        self.data = None
        self.content_layout = None
        self.main_layout = None

    def configure_image_widgets(self):
        self.generate_marginal_for((), 0, 0, 'xy', cursors=False, layout=self.content_layout)
        self.generate_marginal_for((), 1, 0, 'kxy', cursors=False, layout=self.content_layout)

    def add_contextual_widgets(self):
        convert_dims = ['theta', 'beta', 'phi', 'psi']
        if 'eV' not in self.data.dims:
            convert_dims += ['chi']
        if 'hv' in self.data.dims:
            convert_dims += ['hv']

        ui = {}
        with CollectUI(ui):
            controls = tabs(
                ['Controls', horizontal(
                    *[vertical(*[vertical(
                        label(p),
                        numeric_input(self.data.attrs.get(f'{p}_offset', 0.), input_type=float, id=f'control-{p}'),
                    ) for p in pair]) for pair in group_by(2, convert_dims)]
                )]
            )

        def update_dimension_name(dim_name):
            def updater(value):
                self.update_offsets(dict([[dim_name, float(value)]]))
            return updater

        for dim in convert_dims:
            ui[f'control-{dim}'].subject.subscribe(update_dimension_name(dim))

        controls.setFixedHeight(qt_info.inches_to_px(1.75))

        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(controls, 1, 0)

    def update_offsets(self, offsets):
        self.data.S.apply_offsets(offsets)
        self.update_data()

    def layout(self):
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        return self.main_layout

    def update_data(self):
        self.views['xy'].setImage(self.data)

        kdata = convert_to_kspace(self.data, **self.conversion_kwargs)
        if 'eV' in kdata:
            kdata = kdata.S.transpose_to_back('eV')

        self.views['kxy'].setImage(kdata.S.nan_to_num())

    def before_show(self):
        self.configure_image_widgets()
        self.add_contextual_widgets()
        import matplotlib.cm
        self.set_colormap(matplotlib.cm.viridis)

    def after_show(self):
        self.update_data()

    def set_data(self, data: DataType, **kwargs):
        original_data = normalize_to_spectrum(data)
        self.original_data = original_data

        if len(data.dims) > 2:
            assert 'eV' in original_data.dims
            data = data.sel(eV=slice(-0.05, 0.05)).sum('eV', keep_attrs=True)
            data.coords['eV'] = 0
        else:
            data = original_data

        if 'eV' in data.dims:
            data = data.S.transpose_to_back('eV')

        self.data = data.copy(deep=True)

        if not kwargs:
            rng_mul = 1
            if data.coords['hv'] < 12:
                rng_mul = 0.5
            if data.coords['hv'] < 7:
                rng_mul = 0.25

            if 'eV' in self.data.dims:
                kwargs = {'kp': np.linspace(-2, 2, 400) * rng_mul,}
            else:
                kwargs = {'kx': np.linspace(-3, 3, 300) * rng_mul, 'ky': np.linspace(-3, 3, 300) * rng_mul,}

        self.conversion_kwargs = kwargs


def ktool(data: DataType, **kwargs):
    tool = KTool()
    tool.set_data(data, **kwargs)
    tool.start()
