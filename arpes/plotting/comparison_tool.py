from typing import Optional

import xarray as xr
import colorcet as cc
import numpy as np
import scipy.ndimage.interpolation

from arpes.typing import DataType

from arpes.plotting.interactive_utils import BokehInteractiveTool
from arpes.utilities.funcutils import Debounce
from arpes.utilities.normalize import normalize_to_spectrum

__all__ = ['ComparisonTool', 'compare']


class ComparisonTool(BokehInteractiveTool):
    auto_zero_nans = False
    auto_rebin = False
    other = None
    compared = None

    def __init__(self, other, **kwargs):
        super().__init__()

        self.load_settings(**kwargs)

        self.app_main_size = self.settings.get('app_main_size', 600)
        self.other = other

    def tool_handler(self, doc):
        from bokeh.layouts import row, column, widgetbox
        from bokeh.models import widgets
        from bokeh.models.mappers import LinearColorMapper
        from bokeh.plotting import figure

        default_palette = self.default_palette
        difference_palette = cc.coolwarm

        intensity_slider = widgets.Slider(
            title='Relative Intensity Scaling', start=0.5, end=1.5,
            step=0.005, value=1)

        self.app_context.update({
            'A': self.arr,
            'B': self.other,
            'compared': self.compared,
            'plots': {},
            'figures': {},
            'widgets': {},
            'data_range': self.arr.T.range(),
            'color_maps': {},
        })

        self.color_maps['main'] = LinearColorMapper(
            default_palette, low=np.min(self.arr.values), high=np.max(self.arr.values), nan_color='black')

        figure_kwargs = {
            'tools': ['reset', 'wheel_zoom'],
            'plot_width': self.app_main_size,
            'plot_height': self.app_main_size,
            'min_border': 10,
            'toolbar_location': 'left',
            'x_range': self.data_range['x'],
            'y_range': self.data_range['y'],
            'x_axis_location': 'below',
            'y_axis_location': 'right',
        }

        self.figures['A'] = figure(title='Spectrum A', **figure_kwargs)
        self.figures['B'] = figure(title='Spectrum B', **figure_kwargs)
        self.figures['compared'] = figure(title='Comparison', **figure_kwargs)

        self.compared = self.arr - self.other
        diff_low, diff_high = np.min(self.arr.values), np.max(self.arr.values)
        diff_range = np.sqrt((abs(diff_low) + 1) * (abs(diff_high) + 1)) * 1.5
        self.color_maps['difference'] = LinearColorMapper(
            difference_palette, low=-diff_range, high=diff_range, nan_color='white')

        self.plots['A'] = self.figures['A'].image(
            [self.arr.values], x=self.data_range['x'][0], y=self.data_range['y'][0],
            dw=self.data_range['x'][1] - self.data_range['x'][0],
            dh=self.data_range['y'][1] - self.data_range['y'][0],
            color_mapper=self.color_maps['main'],
        )

        self.plots['B'] = self.figures['B'].image(
            [self.other.values], x=self.data_range['x'][0], y=self.data_range['y'][0],
            dw=self.data_range['x'][1] - self.data_range['x'][0],
            dh=self.data_range['y'][1] - self.data_range['y'][0],
            color_mapper=self.color_maps['main']
        )

        self.plots['compared'] = self.figures['compared'].image(
            [self.compared.values], x=self.data_range['x'][0], y=self.data_range['y'][0],
            dw=self.data_range['x'][1] - self.data_range['x'][0],
            dh=self.data_range['y'][1] - self.data_range['y'][0],
            color_mapper=self.color_maps['difference']
        )

        x_axis_name = self.arr.dims[0]
        y_axis_name = self.arr.dims[1]

        stride = self.arr.T.stride()
        delta_x_axis = stride['x']
        delta_y_axis = stride['y']

        delta_x_slider = widgets.Slider(
            title='{} Shift'.format(x_axis_name), start=-20 * delta_x_axis,
            step=delta_x_axis / 2, end=20 * delta_x_axis, value=0)

        delta_y_slider = widgets.Slider(
            title='{} Shift'.format(y_axis_name), start=-20 * delta_y_axis,
            step=delta_y_axis / 2, end=20 * delta_y_axis, value=0)

        @Debounce(0.5)
        def update_summed_figure(attr, old, new):
            # we don't actually use the args because we need to pull all the data out
            shifted = (intensity_slider.value) * scipy.ndimage.interpolation.shift(self.other.values, [
                delta_x_slider.value / delta_x_axis,
                delta_y_slider.value / delta_y_axis,
            ], order=1, prefilter=False, cval=np.nan)
            self.compared = self.arr - xr.DataArray(
                shifted,
                coords=self.arr.coords,
                dims=self.arr.dims)

            self.compared.attrs.update(**self.arr.attrs)
            try:
                del self.compared.attrs['id']
            except KeyError:
                pass

            self.app_context['compared'] = self.compared
            self.plots['compared'].data_source.data = {
                'image': [self.compared.values]
            }

        layout = column(
            row(
                column(self.app_context['figures']['A']),
                column(self.app_context['figures']['B']),
            ),
            row(
                column(self.app_context['figures']['compared']),
                widgetbox(
                    intensity_slider,
                    delta_x_slider,
                    delta_y_slider,
                ),
            )
        )

        update_summed_figure(None, None, None)

        delta_x_slider.on_change('value', update_summed_figure)
        delta_y_slider.on_change('value', update_summed_figure)
        intensity_slider.on_change('value', update_summed_figure)

        doc.add_root(layout)
        doc.title = 'Comparison Tool'


def compare(A: DataType, B: DataType):
    A = normalize_to_spectrum(A)
    attrs = A.attrs
    B = normalize_to_spectrum(B)

    # normalize total intensity
    TOTAL_INTENSITY = 1000000
    A = A / (A.sum(A.dims) / TOTAL_INTENSITY)
    B = B / (B.sum(B.dims) / TOTAL_INTENSITY)
    A.attrs.update(**attrs)

    tool = ComparisonTool(other=B)
    return tool.make_tool(A)