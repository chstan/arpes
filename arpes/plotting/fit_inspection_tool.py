from bokeh import events
import xarray as xr
import numpy as np

from arpes.plotting.interactive_utils import BokehInteractiveTool, CursorTool


__all__ = ('FitCheckTool',)


class FitCheckTool(BokehInteractiveTool, CursorTool):
    """
    Verification of fits
    """

    auto_zero_nans = False
    auto_rebin = False

    def __init__(self, **kwargs):
        super().__init__()

        self.load_settings(**kwargs)
        self.app_main_size = self.settings.get('main_width', 600)
        self.app_marginal_size = self.settings.get('marginal_width', 300)
        self.selected_data = 'data'
        self.use_dataset = True
        self.remove_outliers = True
        self.outlier_clip = 1

    def tool_handler(self, doc):
        from bokeh.layouts import row, column, widgetbox, Spacer
        from bokeh.models.mappers import LinearColorMapper
        from bokeh.models import widgets
        from bokeh.models.widgets.markups import Div
        from bokeh.plotting import figure

        self.arr = self.arr.copy(deep=True)

        if not isinstance(self.arr, xr.Dataset):
            self.use_dataset = False

        residual = None
        if self.use_dataset:
            raw_data = self.arr.data
            raw_data.values[np.isnan(raw_data.values)] = 0
            fit_results = self.arr.results
            residual = self.arr.residual
            residual.values[np.isnan(residual.values)] = 0
        else:
            raw_data = self.arr.attrs['original_data']
            fit_results = self.arr

        fit_direction = [d for d in raw_data.dims if d not in fit_results.dims]
        fit_direction = fit_direction[0]

        two_dimensional = False
        if len(raw_data.dims) != 2:
            two_dimensional = True
            x_coords, y_coords = fit_results.coords[fit_results.dims[0]], fit_results.coords[fit_results.dims[1]]
            z_coords = raw_data.coords[fit_direction]
        else:
            x_coords, y_coords = raw_data.coords[raw_data.dims[0]], raw_data.coords[raw_data.dims[1]]

        if two_dimensional:
            self.settings['palette'] = 'coolwarm'
        default_palette = self.default_palette

        self.app_context.update({
            'data': raw_data,
            'fits': fit_results,
            'residual': residual,
            'original': self.arr,
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
            }
        })
        if two_dimensional:
            self.app_context['data_range']['z'] = (np.min(z_coords.values), np.max(z_coords.values))

        figures, plots, app_widgets = self.app_context['figures'], self.app_context['plots'],\
                                      self.app_context['widgets']

        self.cursor_dims = raw_data.dims
        if two_dimensional:
            self.cursor = [np.mean(self.data_range['x']),
                           np.mean(self.data_range['y']),
                           np.mean(self.data_range['z'])]
        else:
            self.cursor = [np.mean(self.data_range['x']),
                           np.mean(self.data_range['y'])]

        app_widgets['fit_info_div'] = Div(text='')

        self.app_context['color_maps']['main'] = LinearColorMapper(
            default_palette, low=np.min(raw_data.values), high=np.max(raw_data.values), nan_color='black')

        main_tools = ["wheel_zoom", "tap", "reset","save"]
        main_title = 'Fit Inspection Tool: WARNING Unidentified'

        try:
            main_title = 'Fit Inspection Tool: {}'.format(raw_data.S.label[:60])
        except:
            pass

        figures['main'] = figure(
            tools=main_tools, plot_width=self.app_main_size, plot_height=self.app_main_size, min_border=10,
            min_border_left=50,
            toolbar_location='left', x_axis_location='below', y_axis_location='right',
            title=main_title, x_range=self.data_range['x'],
            y_range=self.app_context['data_range']['y'])
        figures['main'].xaxis.axis_label = raw_data.dims[0]
        figures['main'].yaxis.axis_label = raw_data.dims[1]
        figures['main'].toolbar.logo = None
        figures['main'].background_fill_color = "#fafafa"

        data_for_main = raw_data
        if two_dimensional:
            data_for_main = data_for_main.sel(**dict([[fit_direction, self.cursor[2]]]), method='nearest')
        plots['main'] = figures['main'].image(
            [data_for_main.values.T], x=self.app_context['data_range']['x'][0], y=self.app_context['data_range']['y'][0],
            dw=self.app_context['data_range']['x'][1] - self.app_context['data_range']['x'][0],
            dh=self.app_context['data_range']['y'][1] - self.app_context['data_range']['y'][0],
            color_mapper=self.app_context['color_maps']['main'])

        band_centers = [b.center for b in fit_results.F.bands.values()]
        bands_xs = [b.coords[b.dims[0]].values for b in band_centers]
        bands_ys = [b.values for b in band_centers]
        if fit_results.dims[0] == raw_data.dims[1]:
            bands_ys, bands_xs = bands_xs, bands_ys
        plots['band_locations'] = figures['main'].multi_line(
            xs=bands_xs, ys=bands_ys,
            line_color='white', line_width=1, line_dash='dashed')

        # add cursor lines
        cursor_lines = self.add_cursor_lines(figures['main'])

        # marginals
        if not two_dimensional:
            figures['bottom'] = figure(
                plot_width=self.app_main_size, plot_height=self.app_marginal_size, min_border=10,
                title=None, x_range=figures['main'].x_range,
                x_axis_location='above',
                toolbar_location=None, tools=[])
        else:
            figures['bottom'] = Spacer(width=self.app_main_size, height=self.app_marginal_size)

        right_y_range = figures['main'].y_range
        if two_dimensional:
            right_y_range = self.data_range['z']

        figures['right'] = figure(
            plot_width=self.app_marginal_size, plot_height=self.app_main_size, min_border=10,
            title=None, y_range=right_y_range,
            y_axis_location='left',
            toolbar_location=None, tools=[])

        marginal_line_width = 2
        if not two_dimensional:
            bottom_data = raw_data.sel(**dict([[raw_data.dims[1], self.cursor[1]]]), method='nearest')
            right_data = raw_data.sel(**dict([[raw_data.dims[0], self.cursor[0]]]), method='nearest')

            plots['bottom'] = figures['bottom'].line(x=bottom_data.coords[raw_data.dims[0]].values,
                                                     y=bottom_data.values, line_width=marginal_line_width)
            plots['bottom_residual'] = figures['bottom'].line(x=[], y=[], line_color='red',
                                                              line_width=marginal_line_width)
            plots['bottom_fit'] = figures['bottom'].line(
                x=[], y=[], line_color='blue', line_width=marginal_line_width, line_dash='dashed')
            plots['bottom_init_fit'] = figures['bottom'].line(
                x=[], y=[], line_color='green', line_width=marginal_line_width, line_dash='dotted')

            plots['right'] = figures['right'].line(y=right_data.coords[raw_data.dims[1]].values, x=right_data.values,
                                                   line_width=marginal_line_width)
            plots['right_residual'] = figures['right'].line(x=[], y=[], line_color='red',
                                                            line_width=marginal_line_width)
            plots['right_fit'] = figures['right'].line(
                x=[], y=[], line_color='blue', line_width=marginal_line_width, line_dash='dashed')
            plots['right_init_fit'] = figures['right'].line(
                x=[], y=[], line_color='green', line_width=marginal_line_width, line_dash='dotted')
        else:
            right_data = raw_data.sel(**{k: v for k, v in self.cursor_dict.items() if k != fit_direction}, method='nearest')
            plots['right'] = figures['right'].line(y=right_data.coords[right_data.dims[0]].values, x=right_data.values,
                                                   line_width=marginal_line_width)
            plots['right_residual'] = figures['right'].line(x=[], y=[], line_color='red',
                                                            line_width=marginal_line_width)
            plots['right_fit'] = figures['right'].line(
                x=[], y=[], line_color='blue', line_width=marginal_line_width, line_dash='dashed')
            plots['right_init_fit'] = figures['right'].line(
                x=[], y=[], line_color='green', line_width=marginal_line_width, line_dash='dotted')

        def on_change_main_view(attr, old, data_source):
            self.selected_data = data_source
            data = None
            if data_source == 'data':
                data = raw_data.sel(**{k: v for k, v in self.cursor_dict.items() if k == fit_direction},
                                    method='nearest')
            elif data_source == 'residual':
                data = residual.sel(**{k: v for k, v in self.cursor_dict.items() if k == fit_direction},
                                    method='nearest')
            elif two_dimensional:
                data = fit_results.F.s(data_source)
                data.values[np.isnan(data.values)] = 0

            if data is not None:
                if self.remove_outliers:
                    data = data.T.clean_outliers(clip=self.outlier_clip)

                plots['main'].data_source.data = {
                    'image': [data.values.T],
                }
                update_main_colormap(None, None, main_color_range_slider.value)

        def update_fit_display():
            target = 'right'
            if fit_results.dims[0] == raw_data.dims[1]:
                target = 'bottom'

            if two_dimensional:
                target = 'right'
                current_fit = fit_results.sel(**{k: v for k, v in self.cursor_dict.items() if k != fit_direction}, method='nearest').item()
                coord_vals = raw_data.coords[fit_direction].values
            else:
                current_fit = fit_results.sel(**dict([[fit_results.dims[0], self.cursor[0 if target == 'right' else 1]]]), method='nearest').item()
                coord_vals = raw_data.coords[raw_data.dims[0 if target == 'bottom' else 1]].values

            if current_fit is not None:
                app_widgets['fit_info_div'].text = current_fit._repr_html_(short=True)
            else:
                app_widgets['fit_info_div'].text = 'No fit here.'
                plots['{}_residual'.format(target)].data_source.data = {
                    'x': [], 'y': [],
                }
                plots['{}_fit'.format(target)].data_source.data = {
                    'x': [], 'y': [],
                }
                plots['{}_init_fit'.format(target)].data_source.data = {
                    'x': [], 'y': [],
                }
                return

            if target == 'bottom':
                residual_x = coord_vals
                residual_y = current_fit.residual
                init_fit_x = coord_vals
                init_fit_y = current_fit.init_fit
                fit_x = coord_vals
                fit_y = current_fit.best_fit
            else:
                residual_y = coord_vals
                residual_x = current_fit.residual
                init_fit_y = coord_vals
                init_fit_x = current_fit.init_fit
                fit_y = coord_vals
                fit_x = current_fit.best_fit

            plots['{}_residual'.format(target)].data_source.data = {
                'x': residual_x,
                'y': residual_y,
            }
            plots['{}_fit'.format(target)].data_source.data = {
                'x': fit_x,
                'y': fit_y,
            }
            plots['{}_init_fit'.format(target)].data_source.data = {
                'x': init_fit_x,
                'y': init_fit_y,
            }

        def click_right_marginal(event):
            self.cursor = [self.cursor[0], self.cursor[1], event.y]
            on_change_main_view(None, None, self.selected_data)

        def click_main_image(event):
            if two_dimensional:
                self.cursor = [event.x, event.y, self.cursor[2]]
            else:
                self.cursor = [event.x, event.y]

            if not two_dimensional:
                right_marginal_data = raw_data.sel(**dict([[raw_data.dims[0], self.cursor[0]]]), method='nearest')
                bottom_marginal_data = raw_data.sel(**dict([[raw_data.dims[1], self.cursor[1]]]), method='nearest')
                plots['bottom'].data_source.data = {
                    'x': bottom_marginal_data.coords[raw_data.dims[0]].values,
                    'y': bottom_marginal_data.values,
                }
            else:
                right_marginal_data = raw_data.sel(**{k: v for k, v in self.cursor_dict.items() if k != fit_direction}, method='nearest')

            plots['right'].data_source.data = {
                'y': right_marginal_data.coords[right_marginal_data.dims[0]].values,
                'x': right_marginal_data.values,
            }

            update_fit_display()

        def on_change_outlier_clip(attr, old, new):
            self.outlier_clip = new
            on_change_main_view(None, None, self.selected_data)

        def set_remove_outliers(should_remove_outliers):
            if self.remove_outliers != should_remove_outliers:
                self.remove_outliers = should_remove_outliers

                on_change_main_view(None, None, self.selected_data)

        update_main_colormap = self.update_colormap_for('main')
        MAIN_CONTENT_OPTIONS = [
            ('Residual', 'residual'),
            ('Data', 'data'),
        ]

        if two_dimensional:
            available_parameters = fit_results.F.parameter_names

            for param_name in available_parameters:
                MAIN_CONTENT_OPTIONS.append((param_name, param_name,))

        remove_outliers_toggle = widgets.Toggle(label='Remove Outliers', button_type='primary', active=self.remove_outliers)
        remove_outliers_toggle.on_click(set_remove_outliers)

        outlier_clip_slider = widgets.Slider(title='Clip', start=0, end=10, value=self.outlier_clip,
                                             callback_throttle=150, step=0.2)
        outlier_clip_slider.on_change('value', on_change_outlier_clip)

        main_content_select = widgets.Dropdown(label='Main Content', button_type='primary', menu=MAIN_CONTENT_OPTIONS)
        main_content_select.on_change('value', on_change_main_view)

        # Widgety things
        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range')

        # Attach callbacks
        main_color_range_slider.on_change('value', update_main_colormap)
        figures['main'].on_event(events.Tap, click_main_image)
        if two_dimensional:
            figures['right'].on_event(events.Tap, click_right_marginal)

        layout = row(column(figures['main'], figures.get('bottom')),
                     column(figures['right'], app_widgets['fit_info_div']),
                     column(
                         widgetbox(
                             *[widget for widget in [
                                 self._cursor_info,
                                 main_color_range_slider,
                                 main_content_select,
                                 remove_outliers_toggle if two_dimensional else None,
                                 outlier_clip_slider if two_dimensional else None,
                             ] if widget is not None]
                         ),
                     ))

        update_fit_display()

        doc.add_root(layout)
        doc.title = 'Band Tool'

