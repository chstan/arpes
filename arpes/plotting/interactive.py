import copy
import warnings

import holoviews as hv
import numpy as np
import xarray as xr
from bokeh import events, palettes
from bokeh.layouts import row, column, widgetbox, Spacer
from bokeh.models import ColumnDataSource, HoverTool, widgets
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.widgets.markups import Div
from bokeh.plotting import figure
from holoviews import streams
from skimage import exposure

from .interactive_utils import BokehInteractiveTool

from arpes.fits import GStepBModel, ExponentialDecayCModel
from arpes.io import save_dataset

__all__ = ('ImageTool', 'holoview_spectrum', 'autoview',)

# TODO Implement alignment tool

def holoview_spectrum(arr: xr.DataArray):
    assert (len(arr.dims) == 2)  # This only works for a 2D spectrum

    x_axis, y_axis = tuple(arr.dims)

    main_display = hv.Image(arr)
    tap_stream = hv.streams.Tap(source=main_display,
                                x=float(arr.coords[x_axis].mean()),
                                y=float(arr.coords[y_axis].mean()))

    def tap_crosshairs_x(x, y):
        return main_display.sample(**dict([[x_axis, float(y)]]))

    def tap_crosshairs_y(x, y):
        return main_display.sample(**dict([[y_axis, float(x)]]))

    def cross_hair_info(x, y):
        return hv.HLine(float(y)) * hv.VLine(float(x))

    return main_display * hv.DynamicMap(cross_hair_info, streams=[tap_stream]) \
           << hv.DynamicMap(tap_crosshairs_y, streams=[tap_stream]) \
           << hv.DynamicMap(tap_crosshairs_x, streams=[tap_stream])


def autoview(arr: xr.DataArray):
    if (len(arr.dims) == 2):
        return holoview_spectrum(arr)

    print("Unimplemented for spectra with dimensionality %d" % len(arr.dims))


class ImageTool(BokehInteractiveTool):
    def __init__(self, app_main_size=600, app_marginal_size=300):
        super(ImageTool, self).__init__()
        self.app_main_size = app_main_size
        self.app_marginal_size = app_marginal_size

    # TODO select path in image
    def prep_image(self, image_arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.app_context['color_mode'] == 'linear':
                return image_arr.values

            return exposure.equalize_adapthist(image_arr.values, clip_limit=0.03)

    def tool_handler(self, doc):
        if len(self.arr.shape) == 3:
            return self.tool_handler_3d(doc)

        return self.tool_handler_2d(doc)

    def tool_handler_2d(self, doc):
        arr = self.arr
        # Set up the data
        x_coords, y_coords= arr.coords[arr.dims[0]], arr.coords[arr.dims[1]]

        t0 = None
        fit_data = None

        # Styling
        default_palette = palettes.magma(256)
        error_alpha = 0.3
        error_fill = '#3288bd'

        # Application Organization
        self.app_context.update({
            'data': arr,
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
            },
            'show_stat_variation': False,
            'color_mode': 'linear',
        })

        def stats_patch_from_data(data, subsampling_rate=None):
            if subsampling_rate is None:
                subsampling_rate = int(min(data.values.shape[0] / 50, 5))
                if subsampling_rate == 0:
                    subsampling_rate = 1

            x_values = data.coords[data.dims[0]].values[::subsampling_rate]
            values = data.values[::subsampling_rate]
            sq = np.sqrt(values)
            lower, upper = values - sq, values + sq

            return {
                'x': np.append(x_values, x_values[::-1]),
                'y': np.append(lower, upper[::-1]),
            }

        def update_stat_variation(plot_name, data):
            patch_data = stats_patch_from_data(data)
            if plot_name != 'right':  # the right plot is on transposed axes
                plots[plot_name + '_marginal_err'].data_source.data = patch_data
            else:
                plots[plot_name + '_marginal_err'].data_source.data = {
                    'x': patch_data['y'],
                    'y': patch_data['x'],
                }

        figures, plots, app_widgets = self.app_context['figures'], self.app_context['plots'], self.app_context[
            'widgets']
        self.app_context['cursor'] = [np.mean(self.app_context['data_range']['x']),
                                      np.mean(self.app_context['data_range']['y'])] # Try a sensible default

        # create the main inset plot
        main_image = arr
        prepped_main_image = self.prep_image(main_image)
        self.app_context['color_maps']['main'] = LinearColorMapper(
            default_palette, low=np.min(prepped_main_image), high=np.max(prepped_main_image), nan_color='black')

        main_tools = ["wheel_zoom", "tap", "reset"]
        figures['main'] = figure(
            tools=main_tools, plot_width=self.app_main_size, plot_height=self.app_main_size, min_border=10,
            min_border_left=50,
            toolbar_location='left', x_axis_location='below', y_axis_location='right',
            title="Bokeh Tool: %s" % arr.S.label[:60], x_range=self.app_context['data_range']['x'],
            y_range=self.app_context['data_range']['y'])
        figures['main'].xaxis.axis_label = arr.dims[0]
        figures['main'].yaxis.axis_label = arr.dims[1]
        figures['main'].toolbar.logo = None
        figures['main'].background_fill_color = "#fafafa"
        plots['main'] = figures['main'].image(
            [prepped_main_image.T], x=self.app_context['data_range']['x'][0], y=self.app_context['data_range']['y'][0],
            dw=self.app_context['data_range']['x'][1] - self.app_context['data_range']['x'][0],
            dh=self.app_context['data_range']['y'][1] - self.app_context['data_range']['y'][0],
            color_mapper=self.app_context['color_maps']['main'])

        app_widgets['info_div'] = Div(text='', width=self.app_marginal_size, height=100)

        horiz_cursor_x = list(self.app_context['data_range']['x'])
        horiz_cursor_y = [0, 0]
        vert_cursor_x = [0, 0]
        vert_cursor_y = list(self.app_context['data_range']['y'])

        # Create the bottom marginal plot
        bottom_marginal = arr.sel(**dict([[arr.dims[1], self.app_context['cursor'][1]]]), method='nearest')
        figures['bottom_marginal'] = figure(plot_width=self.app_main_size, plot_height=200,
                                            title=None, x_range=figures['main'].x_range,
                                            y_range=(
                                                np.min(bottom_marginal.values), np.max(bottom_marginal.values)),
                                            x_axis_location='above', toolbar_location=None, tools=[])
        plots['bottom_marginal'] = figures['bottom_marginal'].line(
            x=bottom_marginal.coords[arr.dims[0]].values, y=bottom_marginal.values)
        plots['bottom_marginal_err'] = figures['bottom_marginal'].patch(
            x=[], y=[], color=error_fill, fill_alpha=error_alpha, line_color=None)

        # Create the right marginal plot
        right_marginal = arr.sel(**dict([[arr.dims[0], self.app_context['cursor'][0]]]), method='nearest')
        figures['right_marginal'] = figure(plot_width=200, plot_height=self.app_main_size,
                                           title=None, y_range=figures['main'].y_range,
                                           x_range=(np.min(right_marginal.values), np.max(right_marginal.values)),
                                           y_axis_location='left', toolbar_location=None, tools=[])
        plots['right_marginal'] = figures['right_marginal'].line(
            y=right_marginal.coords[arr.dims[1]].values, x=right_marginal.values)
        plots['right_marginal_err'] = figures['right_marginal'].patch(
            x=[], y=[], color=error_fill, fill_alpha=error_alpha, line_color=None)

        def update_cursor(vert_x, horiz_y):
            horiz_y[0] = horiz_y[1] = self.app_context['cursor'][1]
            vert_x[0] = vert_x[1] = self.app_context['cursor'][0]

        update_cursor(vert_cursor_x, horiz_cursor_y)

        cursor_lines = figures['main'].multi_line(xs=[horiz_cursor_x, vert_cursor_x],
                                                  ys=[horiz_cursor_y, vert_cursor_y],
                                                  line_color='white', line_width=2, line_dash='dotted')

        # Attach tools and callbacks
        toggle = widgets.Toggle(label="Show Stat. Variation", button_type="success", active=False)

        def set_show_stat_variation(should_show):
            cursor = self.app_context['cursor']
            self.app_context['show_stat_variation'] = should_show

            if should_show:
                main_image_data = arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
                update_stat_variation('z', arr.sel(**dict([[arr.dims[0], cursor[0]], [arr.dims[1], cursor[1]]]),
                                                   method='nearest'))
                update_stat_variation('bottom',
                                      main_image_data.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest'))
                update_stat_variation('right',
                                      main_image_data.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest'))
                plots['bottom_marginal_err'].visible = True
                plots['right_marginal_err'].visible = True
            else:
                plots['bottom_marginal_err'].visible = False
                plots['right_marginal_err'].visible = False

        toggle.on_click(set_show_stat_variation)

        scan_keys = ['x', 'y', 'z', 'pass_energy', 'hv', 'location', 'id',
                     'probe_pol', 'pump_pol']
        scan_info_source = ColumnDataSource({
            'keys': [k for k in scan_keys if k in arr.attrs],
            'values': [str(v) if isinstance(v, float) and np.isnan(v) else v for v in
                       [arr.attrs[k] for k in scan_keys if k in arr.attrs]],
        })
        scan_info_columns = [
            widgets.TableColumn(field='keys', title='Attr.'),
            widgets.TableColumn(field='values', title='Value'),
        ]

        POINTER_MODES = [
            ('Cursor', 'cursor',),
            ('Path', 'path',),
        ]

        COLOR_MODES = [
            ('Adaptive Hist. Eq. (Slow)', 'adaptive_equalization',),
            # ('Histogram Eq.', 'equalization',), # not implemented
            ('Linear', 'linear',),
            # ('Log', 'log',), # not implemented
        ]

        def on_change_color_mode(attr, old, new_color_mode):
            self.app_context['color_mode'] = new_color_mode
            if old is None or old != new_color_mode:
                cursor = self.app_context['cursor']
                right_image_data = arr.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest')
                bottom_image_data = arr.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest')
                main_image_data = arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
                prepped_right_image = self.prep_image(right_image_data)
                prepped_bottom_image = self.prep_image(bottom_image_data)
                prepped_main_image = self.prep_image(main_image_data)
                plots['right'].data_source.data = {'image': [prepped_right_image]}
                plots['bottom'].data_source.data = {'image': [prepped_bottom_image.T]}
                plots['main'].data_source.data = {'image': [prepped_main_image.T]}
                update_main_colormap(None, None, main_color_range_slider.value)

        color_mode_dropdown = widgets.Dropdown(label='Color Mode', button_type='primary', menu=COLOR_MODES)
        color_mode_dropdown.on_change('value', on_change_color_mode)

        symmetry_point_name_input = widgets.TextInput(title='Symmetry Point Name', value="G")
        snap_checkbox = widgets.CheckboxButtonGroup(labels=['Snap Axes'], active=[])
        place_symmetry_point_at_cursor_button = widgets.Button(label="Place Point", button_type="primary")

        def update_symmetry_points_for_display():
            pass

        def place_symmetry_point():
            cursor_dict = dict(zip(arr.dims, self.app_context['cursor']))
            skip_dimensions = {'eV', 'delay', 'cycle'}
            if 'symmetry_points' not in arr.attrs:
                arr.attrs['symmetry_points'] = {}

            snap_distance = {
                'phi': 2,
                'beta': 2,
                'kx': 0.01,
                'ky': 0.01,
                'kz': 0.01,
                'kp': 0.01,
                'hv': 4,
            }

            cursor_dict = {k: v for k, v in cursor_dict.items() if k not in skip_dimensions}
            snapped = copy.copy(cursor_dict)

            if 'Snap Axes' in [snap_checkbox.labels[i] for i in snap_checkbox.active]:
                for axis, value in cursor_dict.items():
                    options = [point[axis] for point in arr.attrs['symmetry_points'].values() if axis in point]
                    options = sorted(options, key=lambda x: np.abs(x - value))
                    if len(options) and np.abs(options[0] - value) < snap_distance[axis]:
                        snapped[axis] = options[0]

            arr.attrs['symmetry_points'][symmetry_point_name_input.value] = snapped

        place_symmetry_point_at_cursor_button.on_click(place_symmetry_point)

        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range (Main)')
        layout = row(column(figures['main'], figures['bottom_marginal']),
                     column(figures['right_marginal'], Spacer(width=200, height=200)),
                     column(widgetbox(
                         widgets.Dropdown(label='Pointer Mode', button_type='primary', menu=POINTER_MODES)),
                         widgets.Tabs(tabs=[
                             widgets.Panel(child=widgetbox(
                                 Div(text='<h2>Colorscale:</h2>'),
                                 color_mode_dropdown,
                                 main_color_range_slider,
                                 Div(text='<h2 style="padding-top: 30px;">General Settings:</h2>'),
                                 toggle,
                                 sizing_mode='scale_width'
                             ), title='Settings'),
                             widgets.Panel(child=widgetbox(
                                 app_widgets['info_div'],
                                 Div(text='<h2 style="padding-top: 30px; padding-bottom: 10px;">Scan Info</h2>'),
                                 widgets.DataTable(source=scan_info_source, columns=scan_info_columns, width=400,
                                                   height=400),
                                 sizing_mode='scale_width', width=400
                             ), title='Info'),
                             widgets.Panel(child=widgetbox(
                                 Div(text='<h2>Preparation</h2>'),
                                 symmetry_point_name_input,
                                 snap_checkbox,
                                 place_symmetry_point_at_cursor_button,
                                 sizing_mode='scale_width'
                             ), title='Preparation'),
                         ], width=400)))

        def update_colormap_for(plot_name):
            def update_plot_colormap(attr, old, new):
                plot_data = plots[plot_name].data_source.data['image']
                low, high = np.min(plot_data), np.max(plot_data)
                dynamic_range = high - low
                self.app_context['color_maps'][plot_name].update(low=low + new[0] / 100 * dynamic_range,
                                                                 high=low + new[1] / 100 * dynamic_range)

            return update_plot_colormap

        update_main_colormap = update_colormap_for('main')

        def on_click_save(event):
            save_dataset(arr)
            print(event)

        def click_main_image(event):
            cursor = self.app_context['cursor']
            cursor[0] = event.x
            cursor[1] = event.y
            update_cursor(vert_cursor_x, horiz_cursor_y)
            cursor_lines.data_source.data = {
                'xs': [horiz_cursor_x, vert_cursor_x],
                'ys': [horiz_cursor_y, vert_cursor_y],
            }
            right_marginal_data = arr.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest')
            bottom_marginal_data = arr.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest')
            plots['bottom_marginal'].data_source.data = {
                'x': bottom_marginal_data.coords[arr.dims[0]].values,
                'y': bottom_marginal_data.values,
            }
            plots['right_marginal'].data_source.data = {
                'y': right_marginal_data.coords[arr.dims[1]].values,
                'x': right_marginal_data.values,
            }
            if self.app_context['show_stat_variation']:
                update_stat_variation('right', right_marginal_data)
                update_stat_variation('bottom', bottom_marginal_data)
            figures['bottom_marginal'].y_range.start = np.min(bottom_marginal_data.values)
            figures['bottom_marginal'].y_range.end = np.max(bottom_marginal_data.values)
            figures['right_marginal'].x_range.start = np.min(right_marginal_data.values)
            figures['right_marginal'].x_range.end = np.max(right_marginal_data.values)

        figures['main'].on_event(events.Tap, click_main_image)
        main_color_range_slider.on_change('value', update_main_colormap)

        doc.add_root(layout)
        doc.title = "Bokeh Tool"


    def tool_handler_3d(self, doc):
        arr = self.arr
        # Set up the data
        x_coords, y_coords, z_coords = arr.coords[arr.dims[0]], arr.coords[arr.dims[1]], arr.coords[arr.dims[2]]

        t0 = None
        fit_data = None

        info_formatters = {
            'eV': """<div>
                            <h2>Fermi Edge Info:</h2>
                            <p>Gap: <b>{:.1f} meV</b></p>
                            <p>Edge Width: <b>{:.1f} meV</b></p>
                            </div>
                            """,
            'delay': """<div>
                            <h2>Delay Scan Info:</h2>
                            <p>t0: <b>{:.1f} fs</b></p>
                            <p>Decay time: <b>{:.1f} fs</b></p>
                            </div>
                            """
        }

        # Styling
        default_palette = palettes.magma(256)
        error_alpha = 0.3
        error_fill = '#3288bd'

        # Application Organization
        self.app_context.update({
            'data': arr,
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
                'z': (np.min(z_coords.values), np.max(z_coords.values)),
            },
            'show_stat_variation': False,
            'color_mode': 'linear',
        })

        def stats_patch_from_data(data, subsampling_rate=None):
            if subsampling_rate is None:
                subsampling_rate = int(min(data.values.shape[0] / 50, 5))
                if subsampling_rate == 0:
                    subsampling_rate = 1

            x_values = data.coords[data.dims[0]].values[::subsampling_rate]
            values = data.values[::subsampling_rate]
            sq = np.sqrt(values)
            lower, upper = values - sq, values + sq

            return {
                'x': np.append(x_values, x_values[::-1]),
                'y': np.append(lower, upper[::-1]),
            }

        def update_stat_variation(plot_name, data):
            patch_data = stats_patch_from_data(data)
            if plot_name != 'right':  # the right plot is on transposed axes
                plots[plot_name + '_marginal_err'].data_source.data = patch_data
            else:
                plots[plot_name + '_marginal_err'].data_source.data = {
                    'x': patch_data['y'],
                    'y': patch_data['x'],
                }

        figures, plots, app_widgets = self.app_context['figures'], self.app_context['plots'], self.app_context[
            'widgets']
        self.app_context['cursor'] = [np.mean(self.app_context['data_range']['x']),
                                      np.mean(self.app_context['data_range']['y']),
                                      np.mean(self.app_context['data_range']['z'])]  # Try a sensible default

        # create the main inset plot
        main_image = arr.sel(**dict([[arr.dims[2], self.app_context['cursor'][2]]]), method='nearest')
        prepped_main_image = self.prep_image(main_image)
        self.app_context['color_maps']['main'] = LinearColorMapper(
            default_palette, low=np.min(prepped_main_image), high=np.max(prepped_main_image), nan_color='black')

        main_tools = ["wheel_zoom", "tap", "reset"]
        figures['main'] = figure(
            tools=main_tools, plot_width=self.app_main_size, plot_height=self.app_main_size, min_border=10,
            min_border_left=50,
            toolbar_location='left', x_axis_location='below', y_axis_location='right',
            title="Bokeh Tool: %s" % arr.S.label[:60], x_range=self.app_context['data_range']['x'],
            y_range=self.app_context['data_range']['y'])
        figures['main'].xaxis.axis_label = arr.dims[0]
        figures['main'].yaxis.axis_label = arr.dims[1]
        figures['main'].toolbar.logo = None
        figures['main'].background_fill_color = "#fafafa"
        plots['main'] = figures['main'].image(
            [prepped_main_image.T], x=self.app_context['data_range']['x'][0], y=self.app_context['data_range']['y'][0],
            dw=self.app_context['data_range']['x'][1] - self.app_context['data_range']['x'][0],
            dh=self.app_context['data_range']['y'][1] - self.app_context['data_range']['y'][0],
            color_mapper=self.app_context['color_maps']['main'])

        # Create the z-selector
        z_marginal_data = arr.sel(
            **dict([[arr.dims[0], self.app_context['cursor'][0]], [arr.dims[1], self.app_context['cursor'][1]]]),
            method='nearest')
        z_hover_tool = HoverTool(
            tooltips=[
                ('x', '@x{1.111}'),
                ('Int. (arb.)', '@y{1.1}'),
            ],
            mode='vline'
        )
        z_tools = [z_hover_tool, 'wheel_zoom', 'tap', 'reset']
        figures['z_marginal'] = figure(
            plot_width=self.app_marginal_size, plot_height=self.app_marginal_size,
            x_range=self.app_context['data_range']['z'],
            y_range=(np.min(z_marginal_data.values), np.max(z_marginal_data.values)),
            x_axis_location='above', toolbar_location='below', y_axis_location='right', tools=z_tools)
        figures['z_marginal'].xaxis.major_label_text_font_size = '0pt'
        figures['z_marginal'].xaxis.axis_label = arr.dims[2]
        figures['z_marginal'].toolbar.logo = None
        plots['z_marginal'] = figures['z_marginal'].line(x=z_coords.values, y=z_marginal_data.values)
        plots['z_marginal_err'] = figures['z_marginal'].patch(
            x=[], y=[], color=error_fill, fill_alpha=error_alpha, line_color=None)

        info_formatter = info_formatters.get(arr.dims[2], '')
        app_widgets['info_div'] = Div(text='', width=self.app_marginal_size, height=100)

        if arr.dims[2] == 'eV':
            # Try to fit a Fermi edge and display it in the plot
            fit_data = z_marginal_data.sel(eV=slice(-0.25, 0.2))
            z_fit = GStepBModel().guess_fit(fit_data)
            plots['z_fit'] = figures['z_marginal'].line(x=fit_data.coords['eV'].values, y=z_fit.best_fit,
                                                        line_dash='dashed', line_color='red')
            self.app_context['z_model'] = GStepBModel

            app_widgets['info_div'].text = info_formatter.format(
                z_fit.params['center'].value * 1000,
                z_fit.params['width'].value * 1000,
            )

        elif arr.dims[2] == 'delay' and 't0' in arr.attrs:
            # Try to fit a decay constant to the data after t0
            plots['t0_marker'] = figures['z_marginal'].line(
                x=[float(arr.attrs.get('t0')), float(arr.attrs.get('t0'))], y=[0, 1000000], line_color='black',
                line_dash='dashed')
            t0 = float(arr.attrs['t0'])
            self.app_context['z_model'] = ExponentialDecayCModel
            try:
                after_t0 = z_marginal_data.sel(delay=slice(t0 - 0.2, None))
                exp_model = ExponentialDecayCModel()
                z_fit = exp_model.guess_fit(after_t0, params={'t0': {'value': t0}})
                plots['z_fit'] = figures['z_marginal'].line(
                    x=after_t0.coords['delay'].values, y=z_fit.best_fit, line_dash='dashed', line_color='red')
            except Exception as e:
                plots['z_fit'] = figures['z_marginal'].line(x=[], y=[], line_dash='dashed', line_color='red')

        horiz_cursor_x = list(self.app_context['data_range']['x'])
        horiz_cursor_y = [0, 0]
        vert_cursor_x = [0, 0]
        vert_cursor_y = list(self.app_context['data_range']['y'])

        # Create the bottom marginal plot
        bottom_image = arr.sel(**dict([[arr.dims[1], self.app_context['cursor'][1]]]), method='nearest')
        prepped_bottom_image = self.prep_image(bottom_image)
        self.app_context['color_maps']['bottom'] = LinearColorMapper(
            default_palette, low=np.min(prepped_bottom_image), high=np.max(prepped_bottom_image), nan_color='black')
        figures['bottom'] = figure(plot_width=self.app_main_size, plot_height=self.app_marginal_size,
                                   title=None, x_range=figures['main'].x_range,
                                   y_range=figures['z_marginal'].x_range,
                                   x_axis_location='above',
                                   toolbar_location=None, tools=[])
        figures['bottom'].xaxis.major_label_text_font_size = '0pt'
        plots['bottom'] = figures['bottom'].image([prepped_bottom_image.T], x=self.app_context['data_range']['x'][0],
                                                  y=self.app_context['data_range']['z'][0],
                                                  dw=self.app_context['data_range']['x'][1] -
                                                     self.app_context['data_range']['x'][0],
                                                  dh=self.app_context['data_range']['z'][1] -
                                                     self.app_context['data_range']['z'][0],
                                                  color_mapper=self.app_context['color_maps']['bottom'])
        bottom_marginal = bottom_image.sel(**dict([[arr.dims[2], self.app_context['cursor'][2]]]), method='nearest')
        figures['bottom_marginal'] = figure(plot_width=self.app_main_size, plot_height=200,
                                            title=None, x_range=figures['main'].x_range,
                                            y_range=(
                                                np.min(bottom_marginal.values), np.max(bottom_marginal.values)),
                                            x_axis_location='above', toolbar_location=None, tools=[])
        plots['bottom_marginal'] = figures['bottom_marginal'].line(
            x=bottom_marginal.coords[arr.dims[0]].values, y=bottom_marginal.values)
        plots['bottom_marginal_err'] = figures['bottom_marginal'].patch(
            x=[], y=[], color=error_fill, fill_alpha=error_alpha, line_color=None)

        # Create the right marginal plot
        right_image = arr.sel(**dict([[arr.dims[0], self.app_context['cursor'][0]]]), method='nearest')
        prepped_right_image = self.prep_image(right_image)
        self.app_context['color_maps']['right'] = LinearColorMapper(
            default_palette, low=np.min(prepped_right_image), high=np.max(prepped_right_image), nan_color='black')
        figures['right'] = figure(plot_width=self.app_marginal_size, plot_height=self.app_main_size, title=None,
                                  x_range=figures['z_marginal'].x_range, y_range=figures['main'].y_range,
                                  toolbar_location=None, tools=[])
        figures['right'].yaxis.major_label_text_font_size = '0pt'
        plots['right'] = figures['right'].image([prepped_right_image], x=self.app_context['data_range']['z'][0],
                                                y=self.app_context['data_range']['y'][0],
                                                dw=self.app_context['data_range']['z'][1] -
                                                   self.app_context['data_range']['z'][0],
                                                dh=self.app_context['data_range']['y'][1] -
                                                   self.app_context['data_range']['y'][0],
                                                color_mapper=self.app_context['color_maps']['right'])
        right_marginal = right_image.sel(**dict([[arr.dims[2], self.app_context['cursor'][2]]]), method='nearest')
        figures['right_marginal'] = figure(plot_width=200, plot_height=self.app_main_size,
                                           title=None, y_range=figures['main'].y_range,
                                           x_range=(np.min(right_marginal.values), np.max(right_marginal.values)),
                                           y_axis_location='left', toolbar_location=None, tools=[])
        plots['right_marginal'] = figures['right_marginal'].line(
            y=right_marginal.coords[arr.dims[1]].values, x=right_marginal.values)
        plots['right_marginal_err'] = figures['right_marginal'].patch(
            x=[], y=[], color=error_fill, fill_alpha=error_alpha, line_color=None)

        def update_cursor(vert_x, horiz_y):
            horiz_y[0] = horiz_y[1] = self.app_context['cursor'][1]
            vert_x[0] = vert_x[1] = self.app_context['cursor'][0]

        update_cursor(vert_cursor_x, horiz_cursor_y)

        cursor_lines = figures['main'].multi_line(xs=[horiz_cursor_x, vert_cursor_x],
                                                  ys=[horiz_cursor_y, vert_cursor_y],
                                                  line_color='white', line_width=2, line_dash='dotted')

        # Attach tools and callbacks
        toggle = widgets.Toggle(label="Show Stat. Variation", button_type="success", active=False)

        def set_show_stat_variation(should_show):
            cursor = self.app_context['cursor']
            self.app_context['show_stat_variation'] = should_show

            if should_show:
                main_image_data = arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
                update_stat_variation('z', arr.sel(**dict([[arr.dims[0], cursor[0]], [arr.dims[1], cursor[1]]]),
                                                   method='nearest'))
                update_stat_variation('bottom',
                                      main_image_data.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest'))
                update_stat_variation('right',
                                      main_image_data.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest'))
                plots['z_marginal_err'].visible = True
                plots['bottom_marginal_err'].visible = True
                plots['right_marginal_err'].visible = True
            else:
                plots['z_marginal_err'].visible = False
                plots['bottom_marginal_err'].visible = False
                plots['right_marginal_err'].visible = False

        toggle.on_click(set_show_stat_variation)

        scan_keys = ['x', 'y', 'z', 'pass_energy', 'hv', 'location', 'id',
                     'probe_pol', 'pump_pol']
        scan_info_source = ColumnDataSource({
            'keys': [k for k in scan_keys if k in arr.attrs],
            'values': [str(v) if isinstance(v, float) and np.isnan(v) else v for v in
                       [arr.attrs[k] for k in scan_keys if k in arr.attrs]],
        })
        scan_info_columns = [
            widgets.TableColumn(field='keys', title='Attr.'),
            widgets.TableColumn(field='values', title='Value'),
        ]

        POINTER_MODES = [
            ('Cursor', 'cursor',),
            ('Path', 'path',),
        ]

        COLOR_MODES = [
            ('Adaptive Hist. Eq. (Slow)', 'adaptive_equalization',),
            # ('Histogram Eq.', 'equalization',), # not implemented
            ('Linear', 'linear',),
            # ('Log', 'log',), # not implemented
        ]

        def on_change_color_mode(attr, old, new_color_mode):
            self.app_context['color_mode'] = new_color_mode
            if old is None or old != new_color_mode:
                cursor = self.app_context['cursor']
                right_image_data = arr.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest')
                bottom_image_data = arr.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest')
                main_image_data = arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
                prepped_right_image = self.prep_image(right_image_data)
                prepped_bottom_image = self.prep_image(bottom_image_data)
                prepped_main_image = self.prep_image(main_image_data)
                plots['right'].data_source.data = {'image': [prepped_right_image]}
                plots['bottom'].data_source.data = {'image': [prepped_bottom_image.T]}
                plots['main'].data_source.data = {'image': [prepped_main_image.T]}
                update_right_colormap(None, None, right_color_range_slider.value)
                update_bottom_colormap(None, None, bottom_color_range_slider.value)
                update_main_colormap(None, None, main_color_range_slider.value)

        color_mode_dropdown = widgets.Dropdown(label='Color Mode', button_type='primary', menu=COLOR_MODES)
        color_mode_dropdown.on_change('value', on_change_color_mode)

        symmetry_point_name_input = widgets.TextInput(title='Symmetry Point Name', value="G")
        snap_checkbox = widgets.CheckboxButtonGroup(labels=['Snap Axes'], active=[])
        place_symmetry_point_at_cursor_button = widgets.Button(label="Place Point", button_type="primary")

        def update_symmetry_points_for_display():
            pass

        def place_symmetry_point():
            cursor_dict = dict(zip(arr.dims, self.app_context['cursor']))
            skip_dimensions = {'eV', 'delay', 'cycle'}
            if 'symmetry_points' not in arr.attrs:
                arr.attrs['symmetry_points'] = {}

            snap_distance = {
                'phi': 2,
                'beta': 2,
                'kx': 0.01,
                'ky': 0.01,
                'kz': 0.01,
                'kp': 0.01,
                'hv': 4,
            }

            cursor_dict = {k: v for k, v in cursor_dict.items() if k not in skip_dimensions}
            snapped = copy.copy(cursor_dict)

            if 'Snap Axes' in [snap_checkbox.labels[i] for i in snap_checkbox.active]:
                for axis, value in cursor_dict.items():
                    options = [point[axis] for point in arr.attrs['symmetry_points'].values() if axis in point]
                    options = sorted(options, key=lambda x: np.abs(x - value))
                    if len(options) and np.abs(options[0] - value) < snap_distance[axis]:
                        snapped[axis] = options[0]

            arr.attrs['symmetry_points'][symmetry_point_name_input.value] = snapped

        place_symmetry_point_at_cursor_button.on_click(place_symmetry_point)

        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range (Main)')
        right_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range (%s Marginal)' % arr.dims[1])
        bottom_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range (%s Marginal)' % arr.dims[0])
        layout = row(column(figures['main'], figures['bottom'], figures['bottom_marginal']),
                     column(figures['right'], figures['z_marginal'], Spacer(width=self.app_marginal_size, height=200)),
                     column(figures['right_marginal'], Spacer(width=200, height=self.app_marginal_size),
                            Spacer(width=200, height=200)),
                     column(widgetbox(
                         widgets.Dropdown(label='Pointer Mode', button_type='primary', menu=POINTER_MODES)),
                         widgets.Tabs(tabs=[
                             widgets.Panel(child=widgetbox(
                                 Div(text='<h2>Colorscale:</h2>'),
                                 color_mode_dropdown,
                                 main_color_range_slider,
                                 right_color_range_slider,
                                 bottom_color_range_slider,
                                 Div(text='<h2 style="padding-top: 30px;">General Settings:</h2>'),
                                 toggle,
                                 sizing_mode='scale_width'
                             ), title='Settings'),
                             widgets.Panel(child=widgetbox(
                                 app_widgets['info_div'],
                                 Div(text='<h2 style="padding-top: 30px; padding-bottom: 10px;">Scan Info</h2>'),
                                 widgets.DataTable(source=scan_info_source, columns=scan_info_columns, width=400,
                                                   height=400),
                                 sizing_mode='scale_width', width=400
                             ), title='Info'),
                             widgets.Panel(child=widgetbox(
                                 Div(text='<h2>Preparation</h2>'),
                                 symmetry_point_name_input,
                                 snap_checkbox,
                                 place_symmetry_point_at_cursor_button,
                                 sizing_mode='scale_width'
                             ), title='Preparation'),
                         ], width=400)))

        def update_colormap_for(plot_name):
            def update_plot_colormap(attr, old, new):
                plot_data = plots[plot_name].data_source.data['image']
                low, high = np.min(plot_data), np.max(plot_data)
                dynamic_range = high - low
                self.app_context['color_maps'][plot_name].update(low=low + new[0] / 100 * dynamic_range,
                                                                 high=low + new[1] / 100 * dynamic_range)

            return update_plot_colormap

        update_main_colormap = update_colormap_for('main')
        update_bottom_colormap = update_colormap_for('bottom')
        update_right_colormap = update_colormap_for('right')

        def on_click_save(event):
            save_dataset(arr)
            print(event)

        def click_z_marginal(event):
            cursor = self.app_context['cursor']
            cursor[2] = event.x
            main_image = arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
            plots['main'].data_source.data = {
                'image': [self.prep_image(main_image).T]
            }
            update_main_colormap(None, None, main_color_range_slider.value)
            right_marginal_data = main_image.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest')
            bottom_marginal_data = main_image.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest')
            plots['bottom_marginal'].data_source.data = {
                'x': bottom_marginal_data.coords[arr.dims[0]].values,
                'y': bottom_marginal_data.values,
            }
            plots['right_marginal'].data_source.data = {
                'y': right_marginal_data.coords[arr.dims[1]].values,
                'x': right_marginal_data.values,
            }
            if self.app_context['show_stat_variation']:
                update_stat_variation('right', right_marginal_data)
                update_stat_variation('bottom', bottom_marginal_data)
            figures['bottom_marginal'].y_range.start = np.min(bottom_marginal_data.values)
            figures['bottom_marginal'].y_range.end = np.max(bottom_marginal_data.values)
            figures['right_marginal'].x_range.start = np.min(right_marginal_data.values)
            figures['right_marginal'].x_range.end = np.max(right_marginal_data.values)

        def click_main_image(event):
            cursor = self.app_context['cursor']
            cursor[0] = event.x
            cursor[1] = event.y
            update_cursor(vert_cursor_x, horiz_cursor_y)
            cursor_lines.data_source.data = {
                'xs': [horiz_cursor_x, vert_cursor_x],
                'ys': [horiz_cursor_y, vert_cursor_y],
            }
            right_image_data = arr.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest')
            bottom_image_data = arr.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest')
            prepped_right_image = self.prep_image(right_image_data)
            prepped_bottom_image = self.prep_image(bottom_image_data)
            plots['right'].data_source.data = {'image': [prepped_right_image]}
            plots['bottom'].data_source.data = {'image': [prepped_bottom_image.T]}
            update_right_colormap(None, None, right_color_range_slider.value)
            update_bottom_colormap(None, None, bottom_color_range_slider.value)
            right_marginal_data = right_image_data.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
            bottom_marginal_data = bottom_image_data.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
            z_data = arr.sel(**dict([[arr.dims[0], cursor[0]], [arr.dims[1], cursor[1]]]), method='nearest')
            plots['z_marginal'].data_source.data = {
                'x': z_coords.values,
                'y': z_data.values,
            }
            plots['bottom_marginal'].data_source.data = {
                'x': bottom_marginal_data.coords[arr.dims[0]].values,
                'y': bottom_marginal_data.values,
            }
            plots['right_marginal'].data_source.data = {
                'y': right_marginal_data.coords[arr.dims[1]].values,
                'x': right_marginal_data.values,
            }
            if self.app_context['show_stat_variation']:
                update_stat_variation('z', z_data)
                update_stat_variation('right', right_marginal_data)
                update_stat_variation('bottom', bottom_marginal_data)
            figures['z_marginal'].y_range.start = np.min(z_data.values)
            figures['z_marginal'].y_range.end = np.max(z_data.values)
            figures['bottom_marginal'].y_range.start = np.min(bottom_marginal_data.values)
            figures['bottom_marginal'].y_range.end = np.max(bottom_marginal_data.values)
            figures['right_marginal'].x_range.start = np.min(right_marginal_data.values)
            figures['right_marginal'].x_range.end = np.max(right_marginal_data.values)
            if 'z_fit' in plots and arr.dims[2] == 'eV':
                z_fit_data = z_data.sel(eV=slice(-0.25, 0.2))
                new_z_fit = GStepBModel().guess_fit(z_fit_data)
                plots['z_fit'].data_source.data = {
                    'x': z_fit_data.coords['eV'].values,
                    'y': new_z_fit.best_fit,
                }
                app_widgets['info_div'].text = info_formatter.format(
                    new_z_fit.params['center'].value * 1000,
                    new_z_fit.params['width'].value * 1000,
                )
            if 'z_fit' in plots and arr.dims[2] == 'delay':
                try:
                    t0 = float(arr.attrs.get('t0', 0))
                    after_t0 = z_data.sel(delay=slice(t0 - 0.2, None))
                    exp_model = ExponentialDecayCModel()
                    z_fit = exp_model.guess_fit(after_t0, params={'t0': {'value': t0}})
                    plots['z_fit'].data_source.data = {
                        'x': after_t0.coords['delay'],
                        'y': z_fit.best_fit,
                    }
                except Exception as e:
                    plots['z_fit'].data_source.data = {
                        'x': [],
                        'y': [],
                    }

        figures['z_marginal'].on_event(events.Tap, click_z_marginal)
        figures['main'].on_event(events.Tap, click_main_image)
        main_color_range_slider.on_change('value', update_main_colormap)
        bottom_color_range_slider.on_change('value', update_bottom_colormap)
        right_color_range_slider.on_change('value', update_right_colormap)

        doc.add_root(layout)
        doc.title = "Bokeh Tool"
