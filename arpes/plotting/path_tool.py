import numpy as np
import xarray as xr

from arpes.analysis.path import select_along_path
from arpes.plotting.interactive_utils import CursorTool, SaveableTool
from arpes.exceptions import AnalysisError

from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ('PathTool', 'path_tool',)


class PathTool(SaveableTool, CursorTool):
    """
    Tool to allow drawing paths on data, creating selections based on paths,
    and masking regions around paths

    Integrates with the tools in arpes.analysis.path
    """

    auto_zero_nans = False
    auto_rebin = False

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop('name', None))

        self.load_settings(**kwargs)

        self.app_main_size = self.settings.get('app_main_size', 600)
        self.app_marginal_size = self.settings.get('app_main_size', 300)
        self.pointer_mode = 'path'

    def tool_handler(self, doc):
        from bokeh import events
        from bokeh.layouts import row, column, widgetbox
        from bokeh.models.mappers import LinearColorMapper
        from bokeh.models import widgets, warnings
        from bokeh.plotting import figure

        if len(self.arr.shape) != 2:
            raise AnalysisError('Cannot use path tool on non image-like spectra')

        arr = self.arr
        x_coords, y_coords = arr.coords[arr.dims[0]], arr.coords[arr.dims[1]]

        default_palette = self.default_palette

        self.app_context.update({
            'path_options': [],
            'active_path': None,
            'paths': {},
            'data': arr,
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
            },
        })

        self.cursor = [np.mean(self.data_range['x']),
                       np.mean(self.data_range['y'])]

        self.color_maps['main'] = LinearColorMapper(default_palette, low=np.min(arr.values), high=np.max(arr.values),
                                                    nan_color='black')

        main_tools = ["wheel_zoom", "tap", "reset"]
        main_title = 'Path Tool: WARNING Unidentified'

        try:
            main_title = 'Path Tool: {}'.format(arr.S.label[:60])
        except:
            pass

        self.figures['main'] = figure(
            tools=main_tools, plot_width=self.app_main_size, plot_height=self.app_main_size, min_border=10,
            min_border_left=20, toolbar_location='left', x_axis_location='below', y_axis_location='right',
            title=main_title, x_range=self.data_range['x'], y_range=self.data_range['y'],
        )

        self.figures['main'].xaxis.axis_label = arr.dims[0]
        self.figures['main'].yaxis.axis_label = arr.dims[1]

        self.plots['main'] = self.figures['main'].image(
            [np.asarray(arr.values.T)], x=self.data_range['x'][0], y=self.data_range['y'][0],
            dw=self.data_range['x'][1] - self.data_range['x'][0],
            dh=self.data_range['y'][1] - self.data_range['y'][0],
            color_mapper=self.color_maps['main']
        )

        self.plots['paths'] = self.figures['main'].multi_line(xs=[], ys=[], line_color='white', line_width=2)

        self.add_cursor_lines(self.figures['main'])

        def add_point_to_path():
            if self.active_path in self.paths:
                self.paths[self.active_path]['points'].append(list(self.cursor))
                update_path_display()

            self.save_app()

        def click_main_image(event):
            self.cursor = [event.x, event.y]
            if self.pointer_mode == 'path':
                add_point_to_path()

        POINTER_MODES = [
            ('Cursor', 'cursor',),
            ('Path', 'path',),
        ]

        def convert_to_xarray():
            """
            For each of the paths, we will create a dataset which has an index dimension,
            and datavariables for each of the coordinate dimensions
            :return:
            """
            def convert_single_path_to_xarray(points):
                vars = {d: np.array([p[i] for p in points]) for i, d in enumerate(self.arr.dims)}
                coords = {
                    'index': np.array(range(len(points))),
                }
                vars = {k: xr.DataArray(v, coords=coords, dims=['index']) for k, v in vars.items()}
                return xr.Dataset(data_vars=vars, coords=coords)

            return {path['name']: convert_single_path_to_xarray(path['points'])
                    for path in self.paths.values()}

        def select(data=None, **kwargs):
            if data is None:
                data = self.arr

            if len(self.paths) > 1:
                warnings.warn('Only using first path.')

            return select_along_path(path=list(convert_to_xarray().items())[0][1],
                                     data=data, **kwargs)

        self.app_context['to_xarray'] = convert_to_xarray
        self.app_context['select'] = select


        pointer_dropdown = widgets.Dropdown(label='Pointer Mode', button_type='primary', menu=POINTER_MODES)
        self.path_dropdown = widgets.Dropdown(label='Active Path', button_type='primary',
                                              menu=self.path_options)

        path_name_input = widgets.TextInput(placeholder='Path name...')
        add_path_button = widgets.Button(label='Add Path')

        clear_path_button = widgets.Button(label='Clear Path')
        remove_path_button = widgets.Button(label='Remove Path')

        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range')

        def add_path(path_name):
            if path_name not in self.paths:
                self.path_options.append((path_name, path_name,))
                self.path_dropdown.menu = self.path_options
                self.paths[path_name] = {
                    'points': [],
                    'name': path_name,
                }

                if self.active_path is None:
                    self.active_path = path_name

                self.save_app()

        def on_change_active_path(attr, old_path_id, path_id):
            self.debug_text = path_id
            self.app_context['active_path'] = path_id
            self.active_path = path_id
            self.save_app()

        def on_change_pointer_mode(attr, old_pointer_mode, pointer_mode):
            self.app_context['pointer_mode'] = pointer_mode
            self.pointer_mode = pointer_mode
            self.save_app()

        def update_path_display():
            self.plots['paths'].data_source.data = {
                'xs': [[point[0] for point in p['points']] for p in self.paths.values()],
                'ys': [[point[1] for point in p['points']] for p in self.paths.values()],
            }
            self.save_app()

        self.update_path_display = update_path_display

        def on_clear_path():
            if self.active_path in self.paths:
                self.paths[self.active_path]['points'] = []
                update_path_display()

        def on_remove_path():
            if self.active_path in self.paths:
                del self.paths[self.active_path]
                new_path_options = [b for b in self.path_options if b[0] != self.active_path]
                self.path_dropdown.menu = new_path_options
                self.path_options = new_path_options
                self.active_path = None
                update_path_display()

        # Attach callbacks
        main_color_range_slider.on_change('value', self.update_colormap_for('main'))

        self.figures['main'].on_event(events.Tap, click_main_image)
        self.path_dropdown.on_change('value', on_change_active_path)
        pointer_dropdown.on_change('value', on_change_pointer_mode)
        add_path_button.on_click(lambda: add_path(path_name_input.value))
        clear_path_button.on_click(on_clear_path)
        remove_path_button.on_click(on_remove_path)

        layout = row(column(self.figures['main']),
                     column(
                         widgetbox(
                             pointer_dropdown,
                             self.path_dropdown,
                         ),
                         row(
                             path_name_input,
                             add_path_button,
                         ),
                         row(
                             clear_path_button,
                             remove_path_button,
                         ),
                         widgetbox(
                             self._cursor_info,
                             main_color_range_slider,
                         ),
                         self.debug_div,
                     ))

        doc.add_root(layout)
        doc.title = 'Path Tool'
        self.load_app()
        self.save_app()

    def serialize(self):
        return {
            'active_path': self.active_path,
            'path_options': self.path_options,
            'paths': self.paths,
            'cursor': self.cursor,
        }

    def deserialize(self, json_data):
        self.cursor = json_data.get('cursor', [0, 0])

        self.app_context['paths'] = json_data.get('paths', {}) or {}
        self.app_context['path_options'] = json_data.get('path_options', [])

        self.path_dropdown.menu = self.app_context['path_options']

        self.update_path_display()


def path_tool(data: DataType, **kwargs):
    data = normalize_to_spectrum(data)

    tool = PathTool(**kwargs)
    return tool.make_tool(data)
