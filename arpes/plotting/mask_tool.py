import numpy as np

from arpes.analysis.mask import apply_mask
from arpes.plotting.interactive_utils import CursorTool, SaveableTool
from arpes.exceptions import AnalysisError

from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ('MaskTool', 'mask')


class MaskTool(SaveableTool, CursorTool):
    """
    Tool to allow masking data by drawing regions.
    """

    auto_zero_nans = False
    auto_rebin = False

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop('name', None))

        self.load_settings(**kwargs)

        self.app_main_size = self.settings.get('app_main_size', 600)
        self.app_marginal_size = self.settings.get('app_main_size', 300)
        self.pointer_mode = 'region'

    def tool_handler(self, doc):
        from bokeh import events
        from bokeh.layouts import row, column, widgetbox
        from bokeh.models.mappers import LinearColorMapper
        from bokeh.models import widgets
        from bokeh.plotting import figure

        if len(self.arr.shape) != 2:
            raise AnalysisError('Cannot use mask tool on non image-like spectra')

        arr = self.arr
        x_coords, y_coords = arr.coords[arr.dims[0]], arr.coords[arr.dims[1]]

        default_palette = self.default_palette

        self.app_context.update({
            'region_options': [],
            'regions': {},
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
        main_title = 'Mask Tool: WARNING Unidentified'

        try:
            main_title = 'Mask Tool: {}'.format(arr.S.label[:60])
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

        self.add_cursor_lines(self.figures['main'])
        region_patches = self.figures['main'].patches(xs=[], ys=[], color='white', alpha=0.35, line_width=1)

        def add_point_to_region():
            if self.active_region in self.regions:
                self.regions[self.active_region]['points'].append(list(self.cursor))
                update_region_display()

            self.save_app()

        def click_main_image(event):
            self.cursor = [event.x, event.y]
            if self.pointer_mode == 'region':
                add_point_to_region()

        POINTER_MODES = [
            ('Cursor', 'cursor',),
            ('Region', 'region',),
        ]

        def mask(data=None, **kwargs):
            if data is None:
                data = arr

            data = normalize_to_spectrum(data)
            return apply_mask(data, self.app_context['mask'], **kwargs)

        self.app_context['perform_mask'] = mask
        self.app_context['mask'] = None

        pointer_dropdown = widgets.Dropdown(label='Pointer Mode', button_type='primary', menu=POINTER_MODES)
        self.region_dropdown = widgets.Dropdown(label='Active Region', button_type='primary',
                                                menu=self.region_options)

        edge_mask_button = widgets.Button(label='Edge Mask')
        region_name_input = widgets.TextInput(placeholder='Region name...')
        add_region_button = widgets.Button(label='Add Region')

        clear_region_button = widgets.Button(label='Clear Region')
        remove_region_button = widgets.Button(label='Remove Region')

        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range')

        def on_click_edge_mask():
            if self.active_region in self.regions:
                old_points = self.regions[self.active_region]['points']
                dims = [d for d in arr.dims if 'eV' != d]
                energy_index = arr.dims.index('eV')
                max_energy = np.max([p[energy_index] for p in old_points])

                other_dim = dims[0]
                other_coord = arr.coords[other_dim].values
                min_other, max_other = np.min(other_coord), np.max(other_coord)
                min_e = np.min(arr.coords['eV'].values)

                if arr.dims.index('eV') == 0:
                    before = [[min_e - 3, min_other - 1], [0, min_other - 1]]
                    after = [[0, max_other + 1], [min_e - 3, max_other + 1]]
                else:
                    before = [[min_other - 1, min_e - 3], [min_other - 1, 0]]
                    after = [[max_other + 1, 0], [max_other + 1, min_e - 3]]
                self.regions[self.active_region]['points'] = before + old_points + after
                self.app_context['mask'] = self.app_context['mask'] or {}
                self.app_context['mask']['fermi'] = max_energy
                update_region_display()

            self.save_app()

        def add_region(region_name):
            if region_name not in self.regions:
                self.region_options.append((region_name, region_name,))
                self.region_dropdown.menu = self.region_options
                self.regions[region_name] = {
                    'points': [],
                    'name': region_name,
                }

                if self.active_region is None:
                    self.active_region = region_name

                self.save_app()

        def on_change_active_region(attr, old_region_id, region_id):
            self.app_context['active_region'] = region_id
            self.active_region = region_id
            self.save_app()

        def on_change_pointer_mode(attr, old_pointer_mode, pointer_mode):
            self.app_context['pointer_mode'] = pointer_mode
            self.pointer_mode = pointer_mode
            self.save_app()

        def update_region_display():
            region_names = self.regions.keys()

            if self.app_context['mask'] is None:
                self.app_context['mask'] = {}
            self.app_context['mask'].update({
                'dims': arr.dims,
                'polys': [r['points'] for r in self.regions.values()]
            })

            region_patches.data_source.data = {
                'xs': [[p[0] for p in self.regions[r]['points']] for r in region_names],
                'ys': [[p[1] for p in self.regions[r]['points']] for r in region_names],
            }
            self.save_app()

        self.update_region_display = update_region_display

        def on_clear_region():
            if self.active_region in self.regions:
                self.regions[self.active_region]['points'] = []
                update_region_display()

        def on_remove_region():
            if self.active_region in self.regions:
                del self.regions[self.active_region]
                new_region_options = [b for b in self.region_options if b[0] != self.active_region]
                self.region_dropdown.menu = new_region_options
                self.region_options = new_region_options
                self.active_region = None
                update_region_display()

        # Attach callbacks
        main_color_range_slider.on_change('value', self.update_colormap_for('main'))

        self.figures['main'].on_event(events.Tap, click_main_image)
        self.region_dropdown.on_change('value', on_change_active_region)
        pointer_dropdown.on_change('value', on_change_pointer_mode)
        add_region_button.on_click(lambda: add_region(region_name_input.value))
        edge_mask_button.on_click(on_click_edge_mask)
        clear_region_button.on_click(on_clear_region)
        remove_region_button.on_click(on_remove_region)

        layout = row(column(self.figures['main']),
                     column(
                         *[f for f in [
                             widgetbox(
                                 pointer_dropdown,
                                 self.region_dropdown,
                             ),
                             row(
                                 region_name_input,
                                 add_region_button,
                             ),
                             edge_mask_button if 'eV' in arr.dims else None,
                             row(
                                 clear_region_button,
                                 remove_region_button,
                             ),
                             widgetbox(
                                 self._cursor_info,
                                 main_color_range_slider,
                             ),
                         ] if f is not None]
                     ))

        doc.add_root(layout)
        doc.title = 'Mask Tool'
        self.load_app()
        self.save_app()

    def serialize(self):
        return {
            'active_region': self.active_region,
            'region_options': self.region_options,
            'regions': self.regions,
            'cursor': self.cursor,
        }

    def deserialize(self, json_data):
        self.cursor = json_data.get('cursor', [0, 0])

        self.app_context['regions'] = json_data.get('regions', {}) or {}
        self.app_context['region_options'] = json_data.get('region_options', [])

        self.region_dropdown.menu = self.app_context['region_options']

        self.update_region_display()


def mask(data: DataType, **kwargs):
    data = normalize_to_spectrum(data)

    tool = MaskTool(**kwargs)
    return tool.make_tool(data)
