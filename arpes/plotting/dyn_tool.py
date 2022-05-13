"""Allows for making any function of a spectrum into a dynamic tool with Bokeh."""
import inspect

import numpy as np

import typing
from arpes.exceptions import AnalysisError
from arpes.plotting.interactive_utils import BokehInteractiveTool, CursorTool
from arpes.typing import DataType
from arpes.utilities import Debounce, normalize_to_spectrum

__all__ = (
    "DynamicTool",
    "dyn",
)


class DynamicTool(BokehInteractiveTool, CursorTool):
    """Presents a utility to rerun a function with different arguments and see the result of the function."""

    def __init__(self, analysis_fn, widget_specification, **kwargs):
        """Initialize the tool and load settings from the user specified ones."""
        super().__init__()

        self.load_settings(**kwargs)
        self.analysis_fn = analysis_fn
        self.widget_specification = widget_specification

        self.app_main_size = self.settings.get("main_width", 600)
        self.app_marginal_size = self.settings.get("marginal_width", 300)

    def tool_handler(self, doc):
        """Configures widgets for the dynamic tool.

        In order to accomplish this, we need to inspect the type signature
        for the function and generate inputs for it dynamically.
        """
        from bokeh import events
        from bokeh.layouts import row, column
        from bokeh.models.mappers import LinearColorMapper
        from bokeh.models import widgets
        from bokeh.plotting import figure

        if len(self.arr.shape) != 2:
            raise AnalysisError("Cannot use the band tool on non image-like spectra")

        self.data_for_display = self.arr
        x_coords, y_coords = (
            self.data_for_display.coords[self.data_for_display.dims[0]],
            self.data_for_display.coords[self.data_for_display.dims[1]],
        )

        default_palette = self.default_palette

        self.app_context.update(
            {
                "data": self.arr,
                "data_range": {
                    "x": (np.min(x_coords.values), np.max(x_coords.values)),
                    "y": (np.min(y_coords.values), np.max(y_coords.values)),
                },
            }
        )

        figures, plots = self.app_context["figures"], self.app_context["plots"]

        self.cursor = [np.mean(self.data_range["x"]), np.mean(self.data_range["y"])]

        self.color_maps["main"] = LinearColorMapper(
            default_palette,
            low=np.min(self.data_for_display.values),
            high=np.max(self.data_for_display.values),
            nan_color="black",
        )

        main_tools = ["wheel_zoom", "tap", "reset"]
        main_title = "{} Tool: WARNING Unidentified".format(self.analysis_fn.__name__)

        try:
            main_title = "{} Tool: {}".format(
                self.analysis_fn.__name__, self.data_for_display.S.label[:60]
            )
        except:
            pass

        figures["main"] = figure(
            tools=main_tools,
            plot_width=self.app_main_size,
            plot_height=self.app_main_size,
            min_border=10,
            min_border_left=50,
            toolbar_location="left",
            x_axis_location="below",
            y_axis_location="right",
            title=main_title,
            x_range=self.data_range["x"],
            y_range=self.data_range["y"],
        )
        figures["main"].xaxis.axis_label = self.data_for_display.dims[0]
        figures["main"].yaxis.axis_label = self.data_for_display.dims[1]
        figures["main"].toolbar.logo = None
        figures["main"].background_fill_color = "#fafafa"
        plots["main"] = figures["main"].image(
            [self.data_for_display.values.T],
            x=self.app_context["data_range"]["x"][0],
            y=self.app_context["data_range"]["y"][0],
            dw=self.app_context["data_range"]["x"][1] - self.app_context["data_range"]["x"][0],
            dh=self.app_context["data_range"]["y"][1] - self.app_context["data_range"]["y"][0],
            color_mapper=self.app_context["color_maps"]["main"],
        )

        # Create the bottom marginal plot
        bottom_marginal = self.data_for_display.sel(
            **dict([[self.data_for_display.dims[1], self.cursor[1]]]), method="nearest"
        )
        bottom_marginal_original = self.arr.sel(
            **dict([[self.data_for_display.dims[1], self.cursor[1]]]), method="nearest"
        )
        figures["bottom_marginal"] = figure(
            plot_width=self.app_main_size,
            plot_height=200,
            title=None,
            x_range=figures["main"].x_range,
            y_range=(np.min(bottom_marginal.values), np.max(bottom_marginal.values)),
            x_axis_location="above",
            toolbar_location=None,
            tools=[],
        )
        plots["bottom_marginal"] = figures["bottom_marginal"].line(
            x=bottom_marginal.coords[self.data_for_display.dims[0]].values, y=bottom_marginal.values
        )
        plots["bottom_marginal_original"] = figures["bottom_marginal"].line(
            x=bottom_marginal_original.coords[self.arr.dims[0]].values,
            y=bottom_marginal_original.values,
            line_color="red",
        )

        # Create the right marginal plot
        right_marginal = self.data_for_display.sel(
            **dict([[self.data_for_display.dims[0], self.cursor[0]]]), method="nearest"
        )
        right_marginal_original = self.arr.sel(
            **dict([[self.data_for_display.dims[0], self.cursor[0]]]), method="nearest"
        )
        figures["right_marginal"] = figure(
            plot_width=200,
            plot_height=self.app_main_size,
            title=None,
            y_range=figures["main"].y_range,
            x_range=(np.min(right_marginal.values), np.max(right_marginal.values)),
            y_axis_location="left",
            toolbar_location=None,
            tools=[],
        )
        plots["right_marginal"] = figures["right_marginal"].line(
            y=right_marginal.coords[self.data_for_display.dims[1]].values, x=right_marginal.values
        )
        plots["right_marginal_original"] = figures["right_marginal"].line(
            y=right_marginal_original.coords[self.data_for_display.dims[1]].values,
            x=right_marginal_original.values,
            line_color="red",
        )

        # add lines
        self.add_cursor_lines(figures["main"])
        _ = figures["main"].multi_line(xs=[], ys=[], line_color="white", line_width=1)  # band lines

        # prep the widgets for the analysis function
        signature = inspect.signature(self.analysis_fn)

        # drop the first which has to be the input data, we can revisit later if this is too limiting
        parameter_names = list(signature.parameters)[1:]
        named_widgets = dict(zip(parameter_names, self.widget_specification))
        built_widgets = {}

        def update_marginals():
            right_marginal_data = self.data_for_display.sel(
                **dict([[self.data_for_display.dims[0], self.cursor[0]]]), method="nearest"
            )
            bottom_marginal_data = self.data_for_display.sel(
                **dict([[self.data_for_display.dims[1], self.cursor[1]]]), method="nearest"
            )
            plots["bottom_marginal"].data_source.data = {
                "x": bottom_marginal_data.coords[self.data_for_display.dims[0]].values,
                "y": bottom_marginal_data.values,
            }
            plots["right_marginal"].data_source.data = {
                "y": right_marginal_data.coords[self.data_for_display.dims[1]].values,
                "x": right_marginal_data.values,
            }

            right_marginal_data = self.arr.sel(
                **dict([[self.data_for_display.dims[0], self.cursor[0]]]), method="nearest"
            )
            bottom_marginal_data = self.arr.sel(
                **dict([[self.data_for_display.dims[1], self.cursor[1]]]), method="nearest"
            )
            plots["bottom_marginal_original"].data_source.data = {
                "x": bottom_marginal_data.coords[self.data_for_display.dims[0]].values,
                "y": bottom_marginal_data.values,
            }
            plots["right_marginal_original"].data_source.data = {
                "y": right_marginal_data.coords[self.data_for_display.dims[1]].values,
                "x": right_marginal_data.values,
            }
            figures["bottom_marginal"].y_range.start = np.min(bottom_marginal_data.values)
            figures["bottom_marginal"].y_range.end = np.max(bottom_marginal_data.values)
            figures["right_marginal"].x_range.start = np.min(right_marginal_data.values)
            figures["right_marginal"].x_range.end = np.max(right_marginal_data.values)

        def click_main_image(event):
            self.cursor = [event.x, event.y]
            update_marginals()

        error_msg = widgets.Div(text="")

        @Debounce(0.25)
        def update_data_for_display():
            try:
                self.data_for_display = self.analysis_fn(
                    self.arr,
                    *[built_widgets[p].value for p in parameter_names if p in built_widgets]
                )
                error_msg.text = ""
            except Exception as e:
                error_msg.text = "{}".format(e)

            # flush + update
            update_marginals()
            plots["main"].data_source.data = {"image": [self.data_for_display.values.T]}

        def update_data_change_wrapper(attr, old, new):
            if old != new:
                update_data_for_display()

        for parameter_name in named_widgets.keys():
            specification = named_widgets[parameter_name]

            widget = None
            if specification["type"] == int:
                widget = widgets.Slider(
                    start=specification["start"],
                    end=specification["end"],
                    value=specification["value"],
                    title=parameter_name,
                )
            if specification["type"] == float:
                widget = widgets.Slider(
                    start=specification["start"],
                    end=specification["end"],
                    value=specification["value"],
                    step=specification["step"],
                    title=parameter_name,
                )

            if widget is not None:
                built_widgets[parameter_name] = widget
                widget.on_change("value", update_data_change_wrapper)

        update_main_colormap = self.update_colormap_for("main")

        self.app_context["run"] = lambda x: x

        main_color_range_slider = widgets.RangeSlider(
            start=0,
            end=100,
            value=(
                0,
                100,
            ),
            title="Color Range",
        )

        # Attach callbacks
        main_color_range_slider.on_change("value", update_main_colormap)

        figures["main"].on_event(events.Tap, click_main_image)

        layout = row(
            column(figures["main"], figures["bottom_marginal"]),
            column(figures["right_marginal"]),
            column(
                column(*[built_widgets[p] for p in parameter_names if p in built_widgets]),
                column(
                    self._cursor_info,
                    main_color_range_slider,
                    error_msg,
                ),
            ),
        )

        doc.add_root(layout)
        doc.title = "Band Tool"


def dyn(dynamic_function: typing.Callable, data: DataType, widget_specifications=None):
    """Starts the dynamic tool using `dynamic_function` and widgets for each arg."""
    data = normalize_to_spectrum(data)

    tool = DynamicTool(dynamic_function, widget_specifications)
    return tool.make_tool(data)
