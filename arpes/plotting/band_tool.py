from bokeh import events
import numpy as np

from arpes.models import band
from arpes.plotting.interactive_utils import BokehInteractiveTool, CursorTool
from exceptions import AnalysisError

from bokeh.layouts import row, column, widgetbox
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import widgets
from bokeh.plotting import figure

from arpes.analysis.band_analysis import fit_patterned_bands

__all__ = ('BandTool',)


class BandTool(BokehInteractiveTool, CursorTool):
    """
    Two dimensional fitting band tool
    """
    auto_zero_nans = False
    auto_rebin = False

    def __init__(self, **kwargs):
        super().__init__()

        self.load_settings(**kwargs)

        self.app_main_size = self.settings.get('app_main_size', 600)
        self.app_marginal_size = self.settings.get('app_main_size', 300)
        self.active_band = None
        self.pointer_mode = 'band'

    def tool_handler(self, doc):
        if len(self.arr.shape) != 2:
            raise AnalysisError('Cannot use the band tool on non image-like spectra')

        arr = self.arr
        x_coords, y_coords = arr.coords[arr.dims[0]], arr.coords[arr.dims[1]]

        default_palette = self.default_palette

        self.app_context.update({
            'bands': {},
            'center_float': None,
            'data': arr,
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
            },
            'direction_normal': True,
            'fit_mode': 'mdc',
        })

        figures, plots, app_widgets = self.app_context['figures'], self.app_context['plots'],\
                                      self.app_context['widgets']
        self.cursor = [np.mean(self.data_range['x']),
                       np.mean(self.data_range['y'])]

        self.color_maps['main'] = LinearColorMapper(
            default_palette, low=np.min(arr.values), high=np.max(arr.values), nan_color='black')

        main_tools = ["wheel_zoom", "tap", "reset"]
        main_title = 'Band Tool: WARNING Unidentified'

        try:
            main_title = 'Band Tool: {}'.format(arr.S.label[:60])
        except:
            pass

        figures['main'] = figure(
            tools=main_tools, plot_width=self.app_main_size, plot_height=self.app_main_size, min_border=10,
            min_border_left=50,
            toolbar_location='left', x_axis_location='below', y_axis_location='right',
            title=main_title, x_range=self.data_range['x'],
            y_range=self.data_range['y'])
        figures['main'].xaxis.axis_label = arr.dims[0]
        figures['main'].yaxis.axis_label = arr.dims[1]
        figures['main'].toolbar.logo = None
        figures['main'].background_fill_color = "#fafafa"
        plots['main'] = figures['main'].image(
            [arr.values.T], x=self.app_context['data_range']['x'][0], y=self.app_context['data_range']['y'][0],
            dw=self.app_context['data_range']['x'][1] - self.app_context['data_range']['x'][0],
            dh=self.app_context['data_range']['y'][1] - self.app_context['data_range']['y'][0],
            color_mapper=self.app_context['color_maps']['main']
        )

        # add lines
        self.add_cursor_lines(figures['main'])
        band_lines = figures['main'].multi_line(xs=[], ys=[], line_color='white', line_width=1)

        def append_point_to_band():
            cursor = self.cursor
            if self.active_band in self.app_context['bands']:
                self.app_context['bands'][self.active_band]['points'].append(list(cursor))
                update_band_display()

        def click_main_image(event):
            self.cursor = [event.x, event.y]
            if self.pointer_mode == 'band':
                append_point_to_band()

        update_main_colormap = self.update_colormap_for('main')

        POINTER_MODES = [
            ('Cursor', 'cursor',),
            ('Band', 'band',),
        ]

        FIT_MODES = [
            ('EDC', 'edc',),
            ('MDC', 'mdc',),
        ]

        DIRECTIONS = [
            ('From Bottom/Left', 'forward',),
            ('From Top/Right', 'reverse'),
        ]

        BAND_TYPES = [
            ('Lorentzian', 'Lorentzian',),
            ('Voigt', 'Voigt',),
            ('Gaussian', 'Gaussian',)
        ]

        band_classes = {
            'Lorentzian': band.Band,
            'Gaussian': band.BackgroundBand,
            'Voigt': band.VoigtBand,
        }

        self.app_context['band_options'] = []

        def pack_bands():
            packed_bands = {}
            for band_name, band_description in self.app_context['bands'].items():
                if len(band_description['points']) == 0:
                    raise AnalysisError('Band {} is empty.'.format(band_name))

                stray = None
                try:
                    stray = float(band_description['center_float'])
                except (KeyError, ValueError, TypeError):
                    try:
                        stray = float(self.app_context['center_float'])
                    except Exception:
                        pass

                packed_bands[band_name] = {
                    'name': band_name,
                    'band': band_classes.get(band_description['type'], band.Band),
                    'dims': self.arr.dims,
                    'constraints': {
                        'amplitude': {'min': 0},
                    },
                    'points': band_description['points'],
                }

                if stray is not None:
                    packed_bands[band_name]['constraints']['stray'] = stray

            return packed_bands

        def fit(override_data=None):
            packed_bands = pack_bands()
            dims = list(self.arr.dims)
            dims.remove('eV')
            angular_direction = dims[0]
            return fit_patterned_bands(override_data or self.arr, packed_bands,
                                       fit_direction='eV' if self.app_context['fit_mode'] == 'edc' else angular_direction,
                                       direction_normal=self.app_context['direction_normal'])

        self.app_context['pack_bands'] = pack_bands
        self.app_context['fit'] = fit

        pointer_dropdown = widgets.Dropdown(label='Pointer Mode', button_type='primary', menu=POINTER_MODES)
        direction_dropdown = widgets.Dropdown(label='Fit Direction', button_type='primary', menu=DIRECTIONS)
        band_dropdown = widgets.Dropdown(label='Active Band', button_type='primary',
                                         menu=self.app_context['band_options'])
        fit_mode_dropdown = widgets.Dropdown(label='Mode', button_type='primary', menu=FIT_MODES)
        band_type_dropdown = widgets.Dropdown(label='Band Type', button_type='primary', menu=BAND_TYPES)

        band_name_input = widgets.TextInput(placeholder='Band name...')
        center_float = widgets.TextInput(placeholder='Center Constraint')
        center_float_copy = widgets.Button(label='Copy to all...')
        add_band_button = widgets.Button(label='Add Band')

        clear_band_button = widgets.Button(label='Clear Band')
        remove_band_button = widgets.Button(label='Remove Band')

        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range')

        def add_band(band_name):
            if band_name not in self.app_context['bands']:
                self.app_context['band_options'].append((band_name, band_name,))
                band_dropdown.menu = self.app_context['band_options']
                self.app_context['bands'][band_name] = {
                    'type': 'Lorentzian',
                    'points': [],
                    'name': band_name,
                    'center_float': None,
                }

                if self.active_band is None:
                    self.active_band = band_name

        def on_copy_center_float():
            for band in self.app_context['bands'].keys():
                self.app_context['bands'][band]['center_float'] = self.app_context['center_float']

        def on_change_active_band(attr, old_band_id, band_id):
            self.app_context['active_band'] = band_id
            self.active_band = band_id

        def on_change_pointer_mode(attr, old_pointer_mode, pointer_mode):
            self.app_context['pointer_mode'] = pointer_mode
            self.pointer_mode = pointer_mode

        def set_center_float_value(attr, old_value, new_value):
            self.app_context['center_float'] = new_value
            if self.active_band in self.app_context['bands']:
                self.app_context['bands'][self.active_band]['center_float'] = new_value

        def set_fit_direction(attr, old_direction, new_direction):
            self.app_context['direction_normal'] = new_direction == 'forward'

        def set_fit_mode(attr, old_mode, new_mode):
            self.app_context['fit_mode'] = new_mode

        def set_band_type(attr, old_type, new_type):
            if self.active_band in self.app_context['bands']:
                self.app_context['bands'][self.active_band]['type'] = new_type

        def update_band_display():
            band_names = self.app_context['bands'].keys()
            band_lines.data_source.data = {
                'xs': [[p[0] for p in self.app_context['bands'][b]['points']] for b in band_names],
                'ys': [[p[1] for p in self.app_context['bands'][b]['points']] for b in band_names],
            }

        def on_clear_band():
            if self.active_band in self.app_context['bands']:
                self.app_context['bands'][self.active_band]['points'] = []
                update_band_display()

        def on_remove_band():
            if self.active_band in self.app_context['bands']:
                del self.app_context['bands'][self.active_band]
                new_band_options = [b for b in self.app_context['band_options'] if b[0] != self.active_band]
                band_dropdown.menu = new_band_options
                self.app_context['band_options'] = new_band_options
                self.active_band = None
                update_band_display()

        # Attach callbacks
        main_color_range_slider.on_change('value', update_main_colormap)

        figures['main'].on_event(events.Tap, click_main_image)
        band_dropdown.on_change('value', on_change_active_band)
        pointer_dropdown.on_change('value', on_change_pointer_mode)
        add_band_button.on_click(lambda: add_band(band_name_input.value))
        clear_band_button.on_click(on_clear_band)
        remove_band_button.on_click(on_remove_band)
        center_float_copy.on_click(on_copy_center_float)
        center_float.on_change('value', set_center_float_value)
        direction_dropdown.on_change('value', set_fit_direction)
        fit_mode_dropdown.on_change('value', set_fit_mode)
        band_type_dropdown.on_change('value', set_band_type)

        layout = row(column(figures['main']),
                     column(
                         widgetbox(
                             pointer_dropdown,
                             band_dropdown,
                             fit_mode_dropdown,
                             band_type_dropdown,
                             direction_dropdown,
                         ),
                         row(
                             band_name_input,
                             add_band_button,
                         ),
                         row(
                             clear_band_button,
                             remove_band_button,
                         ),
                         row(
                             center_float,
                             center_float_copy
                         ),
                         widgetbox(
                             self._cursor_info,
                             main_color_range_slider,
                         )
                     ))

        doc.add_root(layout)
        doc.title = 'Band Tool'