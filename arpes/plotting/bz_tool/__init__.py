import sys
import warnings

import arpes.config

import numpy as np
import xarray as xr

from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from arpes.utilities.conversion import convert_coordinates
from arpes.utilities.bz_spec import SURFACE_ZONE_DEFINITIONS
from arpes.utilities.image import imread_to_xarray
from arpes.plotting.qt_tool.utils import (
    pretty_key_event, PRETTY_KEYS, KeyBinding, hlayout, vlayout, layout, tabs, combobox
)
from arpes.plotting.utils import imshow_arr
from arpes.plotting.qt_tool.excepthook import patched_excepthook

from .CoordinateOffsetWidget import CoordinateOffsetWidget


class BZToolWindow(QtGui.QMainWindow, QtCore.QObject):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tool = None
        self._old_excepthook = sys.excepthook
        sys.excepthook = patched_excepthook

        self._keyBindings = [
            KeyBinding('Close', [QtCore.Qt.Key_Escape], self.do_close),
        ]

        QtGui.QGuiApplication.installEventFilter(self, self)

    def close(self):
        sys.excepthook = self._old_excepthook
        super().close()

    def do_close(self, event):
        self.close()

    def window_print(self, *args, **kwargs):
        print(*args, **kwargs)

    def eventFilter(self, source, event):
        special_keys = [QtCore.Qt.Key_Down, QtCore.Qt.Key_Up, QtCore.Qt.Key_Left, QtCore.Qt.Key_Right]

        if event.type() in [QtCore.QEvent.KeyPress, QtCore.QEvent.ShortcutOverride]:
            if event.type() != QtCore.QEvent.ShortcutOverride or event.key() in special_keys:
                self.handleKeyPressEvent(event)

        return super().eventFilter(source, event)

    def handleKeyPressEvent(self, event):
        handled = False
        for binding in self._keyBindings:
            for combination in binding.chord:
                # only detect single keypresses for now
                if combination == event.key():
                    handled = True
                    binding.handler(event)

        if not handled:
            if arpes.config.SETTINGS.get('DEBUG', False):
                print(event.key())


class BZTool:
    """
    Implements a Brillouin zone explorer showing the region of momentum
    probed by ARPES.
    """

    def __init__(self):
        self.settings = None
        self.context = {}

        self.content_layout = None
        self.main_layout = None
        self.views = {}
        self.reactive_views = []
        self.current_material = None
        self.cut_line = None

        self.canvas = None
        self.ax = None

    def configure_main_widget(self):
        self.canvas = FigureCanvas(Figure(figsize=(8,8)))
        self.ax = self.canvas.figure.subplots()
        self.content_layout.addWidget(self.canvas, 0, 0)

    def on_change_material(self, value):
        self.current_material = SURFACE_ZONE_DEFINITIONS[value]

        if 'bz_points' in self.current_material:
            bz_points = self.current_material['bz_points']
            try:
                bz_points = bz_points()
            except TypeError:
                pass

            x, y = np.concatenate([bz_points, bz_points[:1]]).T
            self.ax.clear()
            self.ax.plot(x, y, linewidth=2)
            self.ax.set_aspect(1)
        else:
            assert 'image' in self.current_material
            image = self.current_material['image']
            image_waypoints = self.current_material['image_waypoints']
            if not image_waypoints[0]:
                warnings.warn('Missing waypoints for material: {}'.format(self.current_material['name']))
                image_waypoints = [
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                ]

            self.image_waypoints = image_waypoints
            self.image_data = imread_to_xarray(image)

            # the way we are currently using waypoints assumes that the data has orthogonal
            # axes and they are not rotated. This might not be appropriate in all circumstances,
            # but it is how we are assuming things for now

            way_p, way_pp = image_waypoints
            way_p_x, way_p_y, way_p_kx, way_p_ky = way_p
            way_pp_x, way_pp_y, way_pp_kx, way_pp_ky = way_pp

            dx, dy, dkx, dky = ((way_pp_x - way_p_x), (way_pp_y - way_p_y),
                                (way_pp_kx - way_p_kx), (way_pp_ky - way_p_ky))

            self.image_data.coords['y'].values = (self.image_data.coords['y'].values - way_p_x) \
                                                 * (dkx / dx) + way_p_kx
            self.image_data.coords['x'].values = (self.image_data.coords['x'].values - way_p_y) \
                                                 * (-dky / dy) - way_p_ky

            self.ax.clear()
            self.image_data.values = self.image_data.values[::-1,:]
            imshow_arr(self.image_data, ax=self.ax, origin='lower')
            self.ax.scatter([0], [0], s=100, color='green')
            self.ax.set_aspect(1)

        self.ax.figure.canvas.draw()
        self.update_cut()

    @property
    def coordinates(self):
        return {k: np.pi * v.value() / 180 for k, v in self.coordinate_widgets.items()}

    def update_cut(self, *args):
        coords = {
            'phi': np.linspace(-15., 15., 51) * np.pi / 180,
            'hv': 5.93, # FOR NOW
            'eV': 0,
        }

        coords_from_widgets = self.coordinates
        coords.update({k: v for k, v in coords_from_widgets.items() if k not in {'phi', }})

        cut = xr.Dataset(data_vars={}, coords=coords, attrs={
            'work_function': self.current_material.get('work_function', 4.2) or 4.2,
            'inner_potential': self.current_material.get('inner_potential', 10) or 10,
            'phi_offset': coords_from_widgets['phi'],
        })

        kcut = convert_coordinates(cut)

        try:
            self.ax.lines.remove(self.cut_line[0])
        except Exception:
            pass

        self.cut_line = self.ax.plot(kcut.kx.values, kcut.ky.values, self.current_material.get('cut_color', 'green'), linewidth=3)
        self.ax.figure.canvas.draw()

    def construct_coordinate_info_tab(self):
        needed_coordinates = ['phi', 'psi', 'alpha', 'theta', 'beta', 'chi']
        inner_items = [CoordinateOffsetWidget(coordinate_name=coordinate, root=self)
                       for coordinate in needed_coordinates]
        self.coordinate_widgets = dict(zip(needed_coordinates, inner_items))

        for widget in inner_items:
            pass
        return hlayout(*inner_items)

    def construct_sample_info_tab(self):
        material_choice = combobox(
            name='Material Specification',
            items=sorted(SURFACE_ZONE_DEFINITIONS.keys())
        )
        self.material_choice_widget = material_choice
        self.material_choice_widget.currentTextChanged.connect(self.on_change_material)
        inner_items = [material_choice]
        return hlayout(*inner_items)

    def construct_detector_info_tab(self):
        inner_items = []
        return hlayout(*inner_items)

    def construct_general_settings_tab(self):
        inner_items = []
        return hlayout(*inner_items)

    def add_coordinate_control_widgets(self):
        self.coordinate_info_tab = self.construct_coordinate_info_tab()
        self.sample_info_tab = self.construct_sample_info_tab()
        self.detector_info_tab = self.construct_detector_info_tab()
        self.general_settings_tab = self.construct_general_settings_tab()

        self.tabs = tabs(
            ['Coordinates', self.coordinate_info_tab],
            ['Sample Info', self.sample_info_tab],
            ['Detector Type and Settings', self.detector_info_tab],
            #['Beamline Info', self.beamline_info],
            ['General Settings', self.general_settings_tab],
        )

        self.tabs.setFixedHeight(250)
        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(self.tabs, 1, 0)

    def start(self):
        app = QtGui.QApplication([])

        win = BZToolWindow()
        win.resize(1100 - 250,1100)
        win.setWindowTitle('Brillouin Zone Tool')
        cw = QtGui.QWidget()
        win.setCentralWidget(cw)
        self.window = win
        self.window.tool = self

        self.content_layout = QtGui.QGridLayout()
        self.main_layout = QtGui.QGridLayout()

        cw.setLayout(self.main_layout)

        self.configure_main_widget()
        self.add_coordinate_control_widgets()

        self.material_choice_widget.setCurrentText('Td-WTe2')

        self.window.show()

        QtGui.QApplication.instance().exec()


def bz_tool():
    tool = BZTool()
    tool.start()