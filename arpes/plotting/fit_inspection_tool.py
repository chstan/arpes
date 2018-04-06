from bokeh import events, palettes
import numpy as np

from arpes.plotting.interactive_utils import BokehInteractiveTool
from exceptions import AnalysisError

from bokeh.layouts import row, column, widgetbox, Spacer
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import widgets
from bokeh.models.widgets.markups import Div
from bokeh.plotting import figure


__all__ = ('FitCheckTool',)


class FitCheckTool(BokehInteractiveTool):
    """
    Verification of fits
    """

    auto_zero_nans = False
    auto_rebin = False

    def __init__(self, app_main_size=600, app_marginal_size=300):
        super().__init__()
        self.active_band = None
        self.pointer_mode = 'band'
        self.app_main_size = app_main_size
        self.app_marginal_size = app_marginal_size

    def tool_handler(self, doc):
        if not 'original_data' in self.arr.attrs:
            raise AnalysisError('You must provide the data used for the fit in attributes under '
                                'the `original_data` key.')

        raw_data = self.arr.attrs['original_data']
        fit_results = self.arr

        if len(raw_data.dims) != 2:
            raise AnalysisError('Cannot use the FitCheckTool on non image-like spectra for now')

        x_coords, y_coords = raw_data.coords[raw_data.dims[0]], raw_data.coords[raw_data.dims[1]]

        default_palette = palettes.magma(256)

        self.app_context.update({
            'data': raw_data,
            'fits': fit_results,
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
            }
        })

        figures, plots, app_widgets = self.app_context['figures'], self.app_context['plots'],\
                                      self.app_context['widgets']

        self.app_context['cursor'] = [np.mean(self.app_context['data_range']['x']),
                                      np.mean(self.app_context['data_range']['y'])]

        horiz_cursor_x = list(self.app_context['data_range']['x'])
        horiz_cursor_y = [0, 0]
        vert_cursor_x = [0, 0]
        vert_cursor_y = list(self.app_context['data_range']['y'])

        app_widgets['cursor_info_div'] = Div(text='')
        app_widgets['fit_info_div'] = Div(text='')

        def set_cursor_info():
            app_widgets['cursor_info_div'].text = '<h2>Cursor:</h2><span>({})</span>'.format(
                ', '.join("{0:.3f}".format(c) for c in self.app_context['cursor']))

        def update_cursor(vert_x, horiz_y):
            horiz_y[0] = horiz_y[1] = self.app_context['cursor'][1]
            vert_x[0] = vert_x[1] = self.app_context['cursor'][0]
            set_cursor_info()

        update_cursor(vert_cursor_x, horiz_cursor_y)

        self.app_context['color_maps']['main'] = LinearColorMapper(
            default_palette, low=np.min(raw_data.values), high=np.max(raw_data.values), nan_color='black')

        main_tools = ["wheel_zoom", "tap", "reset"]
        main_title = 'Fit Inspection Tool: WARNING Unidentified'

        try:
            main_title = 'Fit Inspection Tool: {}'.format(raw_data.S.label[:60])
        except:
            pass

        figures['main'] = figure(
            tools=main_tools, plot_width=self.app_main_size, plot_height=self.app_main_size, min_border=10,
            min_border_left=50,
            toolbar_location='left', x_axis_location='below', y_axis_location='right',
            title=main_title, x_range=self.app_context['data_range']['x'],
            y_range=self.app_context['data_range']['y'])
        figures['main'].xaxis.axis_label = raw_data.dims[0]
        figures['main'].yaxis.axis_label = raw_data.dims[1]
        figures['main'].toolbar.logo = None
        figures['main'].background_fill_color = "#fafafa"
        plots['main'] = figures['main'].image(
            [raw_data.values.T], x=self.app_context['data_range']['x'][0], y=self.app_context['data_range']['y'][0],
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
        cursor_lines = figures['main'].multi_line(xs=[horiz_cursor_x, vert_cursor_x],
                                                  ys=[horiz_cursor_y, vert_cursor_y],
                                                  line_color='white', line_width=2, line_dash='dotted')
        # marginals
        figures['bottom'] = figure(
            plot_width=self.app_main_size, plot_height=self.app_marginal_size, min_border=10,
            title=None, x_range=figures['main'].x_range,
            x_axis_location='above',
            toolbar_location=None, tools=[])

        figures['right'] = figure(
            plot_width=self.app_marginal_size, plot_height=self.app_main_size, min_border=10,
            title=None, y_range=figures['main'].y_range,
            y_axis_location='left',
            toolbar_location=None, tools=[])

        marginal_line_width = 2
        bottom_data = raw_data.sel(**dict([[raw_data.dims[1], self.app_context['cursor'][1]]]), method='nearest')
        right_data = raw_data.sel(**dict([[raw_data.dims[0], self.app_context['cursor'][0]]]), method='nearest')
        plots['bottom'] = figures['bottom'].line(x=bottom_data.coords[raw_data.dims[0]].values, y=bottom_data.values, line_width=marginal_line_width)
        plots['bottom_residual'] = figures['bottom'].line(x=[], y=[], line_color='red', line_width=marginal_line_width)
        plots['bottom_fit'] = figures['bottom'].line(
            x=[], y=[], line_color='blue', line_width=marginal_line_width, line_dash='dashed')
        plots['bottom_init_fit'] = figures['bottom'].line(
            x=[], y=[], line_color='green', line_width=marginal_line_width, line_dash='dotted')

        plots['right'] = figures['right'].line(y=right_data.coords[raw_data.dims[1]].values, x=right_data.values, line_width=marginal_line_width)
        plots['right_residual'] = figures['right'].line(x=[], y=[], line_color='red', line_width=marginal_line_width)
        plots['right_fit'] = figures['right'].line(
            x=[], y=[], line_color='blue', line_width=marginal_line_width, line_dash='dashed')
        plots['right_init_fit'] = figures['right'].line(
            x=[], y=[], line_color='green', line_width=marginal_line_width, line_dash='dotted')

        def update_fit_display():
            target = 'right'
            if fit_results.dims[0] == raw_data.dims[1]:
                target = 'bottom'

            cursor = self.app_context['cursor']

            current_fit = fit_results.sel(**dict([[fit_results.dims[0], cursor[0 if target == 'right' else 1]]]), method='nearest').item()
            app_widgets['fit_info_div'].text = current_fit._repr_html_(short=True)

            if current_fit is None:
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

            # TODO infer mdc or edc, update correct marginal
            coord_vals = raw_data.coords[raw_data.dims[0 if target == 'bottom' else 1]].values

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

        def click_main_image(event):
            cursor = self.app_context['cursor']
            cursor[0] = event.x
            cursor[1] = event.y
            update_cursor(vert_cursor_x, horiz_cursor_y)

            cursor_lines.data_source.data = {
                'xs': [horiz_cursor_x, vert_cursor_x],
                'ys': [horiz_cursor_y, vert_cursor_y],
            }

            right_marginal_data = raw_data.sel(**dict([[raw_data.dims[0], cursor[0]]]), method='nearest')
            bottom_marginal_data = raw_data.sel(**dict([[raw_data.dims[1], cursor[1]]]), method='nearest')
            plots['bottom'].data_source.data = {
                'x': bottom_marginal_data.coords[raw_data.dims[0]].values,
                'y': bottom_marginal_data.values,
            }

            plots['right'].data_source.data = {
                'y': right_marginal_data.coords[raw_data.dims[1]].values,
                'x': right_marginal_data.values,
            }

            update_fit_display()

        def update_colormap_for(plot_name):
            # TODO refactor out duplicated code
            def update_plot_colormap(attr, old, new):
                plot_data = plots[plot_name].data_source.data['image']
                low, high = np.min(plot_data), np.max(plot_data)
                dynamic_range = high - low
                self.app_context['color_maps'][plot_name].update(low=low + new[0] / 100 * dynamic_range,
                                                                 high=low + new[1] / 100 * dynamic_range)

            return update_plot_colormap

        update_main_colormap = update_colormap_for('main')

        # Widgety things
        main_color_range_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Range')

        # Attach callbacks
        main_color_range_slider.on_change('value', update_main_colormap)
        figures['main'].on_event(events.Tap, click_main_image)


        layout = row(column(figures['main'], figures['bottom']),
                     column(figures['right'], app_widgets['fit_info_div']),
                     column(
                         widgetbox(
                             app_widgets['cursor_info_div'],
                             main_color_range_slider,
                         ),
                     ))

        update_fit_display()

        doc.add_root(layout)
        doc.title = 'Band Tool'


