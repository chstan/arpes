import warnings

import numpy as np
import xarray as xr
from bokeh import palettes
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import show
from bokeh.layouts import row, column, widgetbox
from bokeh.models import widgets
from bokeh.models.mappers import LinearColorMapper
from bokeh.plotting import figure

from arpes.analysis import curvature, d1_along_axis, d2_along_axis, gaussian_filter, boxcar_filter
from arpes.io import load_dataset
from arpes.plotting.interactive import init_bokeh_server
from arpes.utilities import Debounce
from typing import Union

__all__ = ['make_curvature_tool']

def make_curvature_tool(arr: Union[xr.DataArray, str], notebook_url='localhost:8888',
                        notebook_handle=True, **kwargs):
    # TODO implement
    # TODO consider whether common structure with make_bokeh_tool would be useful
    if isinstance(arr, str):
        arr = load_dataset(arr)
        if 'cycle' in arr.dims and len(arr.dims) > 2:
            warnings.warn('Summing over cycle')
            arr = arr.sum('cycle', keep_attrs=True)

    # do not remove NaN values because it will be useful to have them in place
    # for masking while we are taking the curvature

    fn_handler, app_context = curvature_tool(arr.T, **kwargs)
    handler = FunctionHandler(fn_handler)
    app = Application(handler)
    show(app, notebook_url=notebook_url, notebook_handle=notebook_handle)

    return app_context


def curvature_tool(arr: xr.DataArray, app_main_size=400):
    init_bokeh_server()
    app_context = {}

    def curvature_tool_handler(doc):
        default_palette = palettes.viridis(256)

        x_coords, y_coords = arr.coords[arr.dims[1]], arr.coords[arr.dims[0]]
        app_context.update({
            'data': arr,
            'cached_data': {},
            'gamma_cached_data': {},
            'plots': {},
            'data_range': {
                'x': (np.min(x_coords.values), np.max(x_coords.values)),
                'y': (np.min(y_coords.values), np.max(y_coords.values)),
            },
            'figures': {},
            'widgets': {},
            'color_maps': {}})

        app_context['color_maps']['d2'] = LinearColorMapper(
            default_palette, low=np.min(arr.values),
            high=np.max(arr.values), nan_color='black')

        app_context['color_maps']['curvature'] = LinearColorMapper(
            default_palette, low=np.min(arr.values),
            high=np.max(arr.values), nan_color='black')

        app_context['color_maps']['raw'] = LinearColorMapper(
            default_palette, low=np.min(arr.values),
            high=np.max(arr.values), nan_color='black')

        plots, figures, data_range, cached_data, gamma_cached_data = (
            app_context['plots'], app_context['figures'], app_context['data_range'],
            app_context['cached_data'], app_context['gamma_cached_data'],
        )

        cached_data['raw'] = arr.values
        gamma_cached_data['raw'] = arr.values

        main_tools = ['reset', 'wheel_zoom']

        figures['d2'] = figure(
            tools=main_tools, plot_width=app_main_size, plot_height=app_main_size,
            min_border=10, toolbar_location='left', x_axis_location='below',
            x_range=data_range['x'], y_range=data_range['y'],
            y_axis_location='right', title='d2 Spectrum'
        )
        figures['curvature'] = figure(
            tools=['reset', 'wheel_zoom'], plot_width=app_main_size, plot_height=app_main_size,
            y_range=app_context['figures']['d2'].y_range,  # link zooming
            x_range=app_context['figures']['d2'].x_range,
            min_border=10, toolbar_location=None, x_axis_location='below',
            y_axis_location='left', title='Curvature'
        )
        figures['raw'] = figure(
            tools=['reset', 'wheel_zoom'], plot_width=app_main_size, plot_height=app_main_size,
            y_range=app_context['figures']['d2'].y_range,  # link zooming
            x_range=app_context['figures']['d2'].x_range,
            min_border=10, toolbar_location=None, x_axis_location='below',
            y_axis_location='left', title='Raw Image'
        )

        figures['curvature'].yaxis.major_label_text_font_size = '0pt'

        # TODO add support for color mapper
        plots['d2'] = figures['d2'].image(
            [arr.values], x=data_range['x'][0], y=data_range['y'][0],
            dw=data_range['x'][1] - data_range['x'][0], dh=data_range['y'][1] - data_range['y'][0],
            color_mapper=app_context['color_maps']['d2']
        )
        plots['curvature'] = figures['curvature'].image(
            [arr.values], x=data_range['x'][0], y=data_range['y'][0],
            dw=data_range['x'][1] - data_range['x'][0], dh=data_range['y'][1] - data_range['y'][0],
            color_mapper=app_context['color_maps']['curvature']
        )
        plots['raw'] = figures['raw'].image(
            [arr.values], x=data_range['x'][0], y=data_range['y'][0],
            dw=data_range['x'][1] - data_range['x'][0], dh=data_range['y'][1] - data_range['y'][0],
            color_mapper=app_context['color_maps']['raw']
        )

        smoothing_sliders_by_name = {}
        smoothing_sliders = [] # need one for each axis
        axis_resolution = {k: abs(arr.coords[k][1] - arr.coords[k][0]) for k in arr.dims}
        for dim in arr.dims:
            coords = arr.coords[dim]
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
            options=list(arr.dims),
            value='eV' if 'eV' in arr.dims else arr.dims[0], # preference to energy,
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

            filter_size = {d: smoothing_sliders_by_name[d].value for d in arr.dims}
            return filter_factory(filter_size, n_passes)

        @Debounce(0.25)
        def force_update():
            n_smoothing_steps = n_smoothing_steps_slider.value
            d2_data = arr
            if interleave_smoothing_toggle.active:
                f = smoothing_fn(n_smoothing_steps // 2)
                d2_data = d1_along_axis(f(d2_data), direction_select.value)
                f = smoothing_fn(n_smoothing_steps - (n_smoothing_steps // 2))
                d2_data = d1_along_axis(f(d2_data), direction_select.value)

            else:
                f = smoothing_fn(n_smoothing_steps)
                d2_data = d2_along_axis(f(arr), direction_select.value)

            if clamp_spectrum_toggle.active:
                d2_data.values = -d2_data.values
                d2_data.values[d2_data.values < 0] = 0
            cached_data['d2'] = d2_data.values
            gamma_cached_data['d2'] = d2_data.values ** gamma_slider.value
            plots['d2'].data_source.data = {
                'image': [gamma_cached_data['d2']]
            }

            curv_smoothing_fn = smoothing_fn(n_smoothing_steps)
            smoothed_curvature_data = curv_smoothing_fn(arr)
            curvature_data = curvature(smoothed_curvature_data, arr.dims, beta=beta_slider.value)
            if clamp_spectrum_toggle.active:
                curvature_data.values = -curvature_data.values
                curvature_data.values[curvature_data.values < 0] = 0

            cached_data['curvature'] = curvature_data.values
            gamma_cached_data['curvature'] = curvature_data.values ** gamma_slider.value
            plots['curvature'].data_source.data = {
                'image': [gamma_cached_data['curvature']]
            }
            update_color_slider(color_slider.value)

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
                app_context['color_maps'][name].update(
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
                column(app_context['figures']['d2'],
                       interleave_smoothing_toggle,
                       direction_select),
                column(app_context['figures']['curvature'],
                       beta_slider, clamp_spectrum_toggle),
                column(app_context['figures']['raw'],
                       color_slider, gamma_slider)
            ),
            widgetbox(
                filter_select,
                *smoothing_sliders,
                n_smoothing_steps_slider,
            )
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

    return curvature_tool_handler, app_context