import holoviews as hv
import numpy as np
import xarray as xr
from bokeh import events, palettes
from bokeh.io import output_notebook
from bokeh.layouts import row, column
from bokeh.plotting import figure
from holoviews import streams

import arpes.config


def init_bokeh_server():
    if 'bokeh_configured' not in arpes.config.CONFIG:
        arpes.config.CONFIG['bokeh_configured'] = True
        output_notebook(hide_banner=True)

        # Don't need to manually start a server in the manner of
        # https://matthewrocklin.com/blog//work/2017/06/28/simple-bokeh-server
        # according to
        # https://github.com/bokeh/bokeh/blob/0.12.10/examples/howto/server_embed/notebook_embed.ipynb

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


def bokeh_tool(arr: xr.DataArray, app_main_size=600, app_marginal_size=300):
    init_bokeh_server()

    def make_bokeh_tool(doc):
        TOOLS = "crosshair,wheel_zoom,tap,reset"

        # Set up the data
        x_coords = arr.coords[arr.dims[0]]
        y_coords = arr.coords[arr.dims[1]]
        z_coords = arr.coords[arr.dims[2]]

        x_rng = (np.min(x_coords.values), np.max(x_coords.values))
        y_rng = (np.min(y_coords.values), np.max(y_coords.values))
        z_rng = (np.min(z_coords.values), np.max(z_coords.values))


        data_name = arr.attrs.get('description', arr.attrs.get('scan', arr.attrs.get('file', 'No Scan Name')))

        cursor = [np.mean(x_rng), np.mean(y_rng), np.mean(z_rng)] # Try a sensible default

        # create the main inset plot
        main_image = arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest')
        main_plot = figure(tools=TOOLS, plot_width=app_main_size, plot_height=app_main_size,
                           min_border=10, min_border_left=50,
                   toolbar_location='above', x_axis_location='below', y_axis_location='right',
                   title="Bokeh Tool: %s" % data_name[:35], x_range=x_rng, y_range=y_rng)
        main_plot.toolbar.logo = None

        main_plot.background_fill_color = "#fafafa"
        main_inset = main_plot.image([main_image.values.T], x=x_rng[0], y=y_rng[0],
                                     dw=x_rng[1] - x_rng[0], dh=y_rng[1] - y_rng[0], palette=palettes.viridis(256))

        # Create the z-selector
        z_marginal_data = arr.sel(**dict([[arr.dims[0], cursor[0]], [arr.dims[1], cursor[1]]]), method='nearest')
        z_selector = figure(plot_width=app_marginal_size, plot_height=app_marginal_size,
                            x_range=z_rng, y_range=(np.min(z_marginal_data.values), np.max(z_marginal_data.values)),
                            toolbar_location='right', tools='crosshair,wheel_zoom,tap,reset')
        z_selector.toolbar.logo = None

        z_inset = z_selector.line(x=z_coords.values, y=z_marginal_data.values)

        horiz_cursor_x = list(x_rng)
        horiz_cursor_y = [0, 0]
        vert_cursor_x = [0, 0]
        vert_cursor_y = list(y_rng)

        # Create the bottom marginal plot
        bottom_image = arr.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest')
        bottom_plot = figure(plot_width=app_main_size, plot_height=app_marginal_size,
                             title='test2', x_range=main_plot.x_range, y_range=z_selector.x_range,
                             toolbar_location=None, tools=[])
        bottom_inset = bottom_plot.image([bottom_image.values.T], x=x_rng[0], y=z_rng[0],
                                         dw=x_rng[1] - x_rng[0], dh=z_rng[1] - z_rng[0], palette=palettes.viridis(256))


        # Create the right marginal plot
        right_image = arr.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest')
        right_plot = figure(plot_width=app_marginal_size, plot_height=app_main_size, title='test',
                            x_range=z_selector.x_range, y_range=main_plot.y_range,
                            toolbar_location=None, tools=[])
        right_inset = right_plot.image([right_image.values], x=z_rng[0], y=y_rng[0],
                                       dw=z_rng[1] - z_rng[0], dh=y_rng[1] - y_rng[0], palette=palettes.viridis(256))



        def update_cursor(vert_x, horiz_y, c):
            horiz_y[0] = horiz_y[1] = c[1]
            vert_x[0] = vert_x[1] = c[0]

        update_cursor(vert_cursor_x, horiz_cursor_y, cursor)

        cursor_lines = main_plot.multi_line(xs=[horiz_cursor_x, vert_cursor_x], ys=[horiz_cursor_y, vert_cursor_y],
                                    line_color='white', line_width=2)

        layout = column(row(main_plot, right_plot),
                        row(bottom_plot, z_selector))

        def click_z_selector(event):
            cursor[2] = event.x
            main_inset.data_source.data = {
                'image': [arr.sel(**dict([[arr.dims[2], cursor[2]]]), method='nearest').values.T]
            }

        def click_main_image(event):
            cursor[0] = event.x
            cursor[1] = event.y
            update_cursor(vert_cursor_x, horiz_cursor_y, cursor)
            cursor_lines.data_source.data = {
                'xs': [horiz_cursor_x, vert_cursor_x],
                'ys': [horiz_cursor_y, vert_cursor_y],
            }
            right_inset.data_source.data = {
                'image': [arr.sel(**dict([[arr.dims[0], cursor[0]]]), method='nearest').values],
            }
            bottom_inset.data_source.data = {
                'image': [arr.sel(**dict([[arr.dims[1], cursor[1]]]), method='nearest').values.T],
            }
            z_data = arr.sel(**dict([[arr.dims[0], cursor[0]], [arr.dims[1], cursor[1]]]), method='nearest').values
            z_inset.data_source.data = {
                'x': z_coords.values,
                'y': z_data,
            }
            z_selector.y_range.start = np.min(z_data)
            z_selector.y_range.end = np.max(z_data)

        z_selector.on_event(events.Tap, click_z_selector)
        main_plot.on_event(events.Tap, click_main_image)

        doc.add_root(layout)
        doc.title = "Bokeh Tool"


    return make_bokeh_tool