import numpy as np

from arpes.analysis import curvature, d1_along_axis, d2_along_axis, gaussian_filter, boxcar_filter
from arpes.plotting.interactive_utils import BokehInteractiveTool
from arpes.utilities.funcutils import Debounce

__all__ = ['CurvatureTool']

class CurvatureTool(BokehInteractiveTool):
    # do not remove nans because they are used for masking in taking derivatives and smoothing
    auto_zero_nans = False
    auto_rebin = False

    def __init__(self, **kwargs):
        super().__init__()

        self.load_settings(**kwargs)
        self.app_main_size = self.settings.get('app_main_size', 400)

    def tool_handler(self, doc):
        from bokeh.layouts import row, column, widgetbox
        from bokeh.models import widgets, Spacer
        from bokeh.models.mappers import LinearColorMapper
        from bokeh.plotting import figure

        default_palette = self.default_palette

        x_coords, y_coords = self.arr.coords[self.arr.dims[1]], self.arr.coords[self.arr.dims[0]]
        self.app_context.update({
            'data': self.arr,
            'cached_data': {},
            'gamma_cached_data': {},
            'plots': {},
            'data_range': self.arr.T.range(),
            'figures': {},
            'widgets': {},
            'color_maps': {}})

        self.app_context['color_maps']['d2'] = LinearColorMapper(
            default_palette, low=np.min(self.arr.values),
            high=np.max(self.arr.values), nan_color='black')

        self.app_context['color_maps']['curvature'] = LinearColorMapper(
            default_palette, low=np.min(self.arr.values),
            high=np.max(self.arr.values), nan_color='black')

        self.app_context['color_maps']['raw'] = LinearColorMapper(
            default_palette, low=np.min(self.arr.values),
            high=np.max(self.arr.values), nan_color='black')

        plots, figures, data_range, cached_data, gamma_cached_data = (
            self.app_context['plots'], self.app_context['figures'], self.app_context['data_range'],
            self.app_context['cached_data'], self.app_context['gamma_cached_data'],
        )

        cached_data['raw'] = self.arr.values
        gamma_cached_data['raw'] = self.arr.values

        figure_kwargs = {
            'tools': ['reset', 'wheel_zoom'],
            'plot_width': self.app_main_size,
            'plot_height': self.app_main_size,
            'min_border': 10,
            'toolbar_location': 'left',
            'x_range': data_range['x'],
            'y_range': data_range['y'],
            'x_axis_location': 'below',
            'y_axis_location': 'right',
        }
        figures['d2'] = figure(title='d2 Spectrum', **figure_kwargs)

        figure_kwargs.update({
            'y_range': self.app_context['figures']['d2'].y_range,
            'x_range': self.app_context['figures']['d2'].x_range,
            'toolbar_location': None, 'y_axis_location': 'left',
        })

        figures['curvature'] = figure(title='Curvature', **figure_kwargs)
        figures['raw'] = figure(title='Raw Image', **figure_kwargs)

        figures['curvature'].yaxis.major_label_text_font_size = '0pt'

        # TODO add support for color mapper
        plots['d2'] = figures['d2'].image(
            [self.arr.values], x=data_range['x'][0], y=data_range['y'][0],
            dw=data_range['x'][1] - data_range['x'][0], dh=data_range['y'][1] - data_range['y'][0],
            color_mapper=self.app_context['color_maps']['d2']
        )
        plots['curvature'] = figures['curvature'].image(
            [self.arr.values], x=data_range['x'][0], y=data_range['y'][0],
            dw=data_range['x'][1] - data_range['x'][0], dh=data_range['y'][1] - data_range['y'][0],
            color_mapper=self.app_context['color_maps']['curvature']
        )
        plots['raw'] = figures['raw'].image(
            [self.arr.values], x=data_range['x'][0], y=data_range['y'][0],
            dw=data_range['x'][1] - data_range['x'][0], dh=data_range['y'][1] - data_range['y'][0],
            color_mapper=self.app_context['color_maps']['raw']
        )

        smoothing_sliders_by_name = {}
        smoothing_sliders = []  # need one for each axis
        axis_resolution = self.arr.T.stride(generic_dim_names=False)
        for dim in self.arr.dims:
            coords = self.arr.coords[dim]
            resolution = float(axis_resolution[dim])
            high_resolution = len(coords) / 3 * resolution
            low_resolution = resolution

            # could make this axis dependent for more reasonable defaults
            default = 15 * resolution

            if default > high_resolution:
                default = (high_resolution + low_resolution) / 2

            new_slider = widgets.Slider(
                title='{} Window'.format(dim), start=low_resolution, end=high_resolution,
                step=resolution, value=default)
            smoothing_sliders.append(new_slider)
            smoothing_sliders_by_name[dim] = new_slider

        n_smoothing_steps_slider = widgets.Slider(
            title="Smoothing Steps", start=0, end=5, step=1, value=2)
        beta_slider = widgets.Slider(
            title="β", start=-8, end=8, step=1, value=0)
        direction_select = widgets.Select(
            options=list(self.arr.dims),
            value='eV' if 'eV' in self.arr.dims else self.arr.dims[0],  # preference to energy,
            title='Derivative Direction'
        )
        interleave_smoothing_toggle = widgets.Toggle(
            label='Interleave smoothing with d/dx', active=True,
            button_type='primary')
        clamp_spectrum_toggle = widgets.Toggle(
            label='Clamp positive values to 0', active=True,
            button_type='primary')
        filter_select = widgets.Select(
            options=['Gaussian', 'Boxcar'],
            value='Boxcar',
            title='Type of Filter'
        )

        color_slider = widgets.RangeSlider(
            start=0, end=100, value=(0, 100,), title='Color Clip')
        gamma_slider = widgets.Slider(
            start=0.1, end=4, value=1, step=0.1, title='Gamma')

        # don't need any cacheing here for now, might if this ends up being too slow
        def smoothing_fn(n_passes):
            if n_passes == 0:
                return lambda x: x

            filter_factory = {
                'Gaussian': gaussian_filter,
                'Boxcar': boxcar_filter,
            }.get(filter_select.value, boxcar_filter)

            filter_size = {d: smoothing_sliders_by_name[d].value for d in self.arr.dims}
            return filter_factory(filter_size, n_passes)

        @Debounce(0.25)
        def force_update():
            n_smoothing_steps = n_smoothing_steps_slider.value
            d2_data = self.arr
            if interleave_smoothing_toggle.active:
                f = smoothing_fn(n_smoothing_steps // 2)
                d2_data = d1_along_axis(f(d2_data), direction_select.value)
                f = smoothing_fn(n_smoothing_steps - (n_smoothing_steps // 2))
                d2_data = d1_along_axis(f(d2_data), direction_select.value)

            else:
                f = smoothing_fn(n_smoothing_steps)
                d2_data = d2_along_axis(f(self.arr), direction_select.value)

            d2_data.values[
                d2_data.values != d2_data.values] = 0  # remove NaN values until Bokeh fixes NaNs over the wire
            if clamp_spectrum_toggle.active:
                d2_data.values = -d2_data.values
                d2_data.values[d2_data.values < 0] = 0
            cached_data['d2'] = d2_data.values
            gamma_cached_data['d2'] = d2_data.values ** gamma_slider.value
            plots['d2'].data_source.data = {
                'image': [gamma_cached_data['d2']]
            }

            curv_smoothing_fn = smoothing_fn(n_smoothing_steps)
            smoothed_curvature_data = curv_smoothing_fn(self.arr)
            curvature_data = curvature(smoothed_curvature_data, self.arr.dims, beta=beta_slider.value)
            curvature_data.values[curvature_data.values != curvature_data.values] = 0
            if clamp_spectrum_toggle.active:
                curvature_data.values = -curvature_data.values
                curvature_data.values[curvature_data.values < 0] = 0

            cached_data['curvature'] = curvature_data.values
            gamma_cached_data['curvature'] = curvature_data.values ** gamma_slider.value
            plots['curvature'].data_source.data = {
                'image': [gamma_cached_data['curvature']]
            }
            update_color_slider(color_slider.value)

        # TODO better integrate these, they can share code with the above if we are more careful.
        def take_d2(d2_data):
            n_smoothing_steps = n_smoothing_steps_slider.value
            if interleave_smoothing_toggle.active:
                f = smoothing_fn(n_smoothing_steps // 2)
                d2_data = d1_along_axis(f(d2_data), direction_select.value)
                f = smoothing_fn(n_smoothing_steps - (n_smoothing_steps // 2))
                d2_data = d1_along_axis(f(d2_data), direction_select.value)

            else:
                f = smoothing_fn(n_smoothing_steps)
                d2_data = d2_along_axis(f(self.arr), direction_select.value)

            d2_data.values[
                d2_data.values != d2_data.values] = 0  # remove NaN values until Bokeh fixes NaNs over the wire
            if clamp_spectrum_toggle.active:
                d2_data.values = -d2_data.values
                d2_data.values[d2_data.values < 0] = 0

            return d2_data

        def take_curvature(curvature_data, curve_dims):
            curv_smoothing_fn = smoothing_fn(n_smoothing_steps_slider.value)
            smoothed_curvature_data = curv_smoothing_fn(curvature_data)
            curvature_data = curvature(smoothed_curvature_data, curve_dims, beta=beta_slider.value)
            curvature_data.values[curvature_data.values != curvature_data.values] = 0
            if clamp_spectrum_toggle.active:
                curvature_data.values = -curvature_data.values
                curvature_data.values[curvature_data.values < 0] = 0

            return curvature_data

        # These functions will always be linked to the current context of the curvature tool.
        self.app_context['d2_fn'] = take_d2
        self.app_context['curvature_fn'] = take_curvature

        def force_update_change_wrapper(attr, old, new):
            if old != new:
                force_update()

        def force_update_click_wrapper(event):
            force_update()

        @Debounce(0.1)
        def update_color_slider(new):
            def update_plot(name, data):
                low, high = np.min(data), np.max(data)
                dynamic_range = high - low
                self.app_context['color_maps'][name].update(
                    low=low + new[0] / 100 * dynamic_range, high=low + new[1] / 100 * dynamic_range)

            update_plot('d2', gamma_cached_data['d2'])
            update_plot('curvature', gamma_cached_data['curvature'])
            update_plot('raw', gamma_cached_data['raw'])

        @Debounce(0.1)
        def update_gamma_slider(new):
            gamma_cached_data['d2'] = cached_data['d2'] ** new
            gamma_cached_data['curvature'] = cached_data['curvature'] ** new
            gamma_cached_data['raw'] = cached_data['raw'] ** new
            update_color_slider(color_slider.value)

        def update_color_handler(attr, old, new):
            update_color_slider(new)

        def update_gamma_handler(attr, old, new):
            update_gamma_slider(new)

        layout = column(
            row(
                column(self.app_context['figures']['d2'],
                       interleave_smoothing_toggle,
                       direction_select),
                column(self.app_context['figures']['curvature'],
                       beta_slider, clamp_spectrum_toggle),
                column(self.app_context['figures']['raw'],
                       color_slider, gamma_slider)
            ),
            widgetbox(
                filter_select,
                *smoothing_sliders,
                n_smoothing_steps_slider,
            ),
            Spacer(height=100),
        )

        # Attach event handlers
        for w in (n_smoothing_steps_slider, beta_slider, direction_select,
                  *smoothing_sliders, filter_select):
            w.on_change('value', force_update_change_wrapper)

        interleave_smoothing_toggle.on_click(force_update_click_wrapper)
        clamp_spectrum_toggle.on_click(force_update_click_wrapper)

        color_slider.on_change('value', update_color_handler)
        gamma_slider.on_change('value', update_gamma_handler)

        force_update()

        doc.add_root(layout)
        doc.title = 'Curvature Tool'