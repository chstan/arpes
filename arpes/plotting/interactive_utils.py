"""Infrastructure code for interactive Bokeh based analysis tools."""
import json
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

import arpes.config
import colorcet as cc
import xarray as xr
from arpes.analysis.general import rebin
from arpes.io import load_data
from arpes.utilities import deep_equals
from typing import List, Union

__all__ = (
    "BokehInteractiveTool",
    "CursorTool",
)


class CursorTool:
    """A base class for a Bokeh based analysis application which using a cursor into a data volume."""

    _cursor = None
    _cursor_info = None
    _horiz_cursor_x = None
    _horiz_cursor_y = None
    _vert_cursor_x = None
    _vert_cursor_y = None
    _cursor_lines = None
    _cursor_dims = None

    def __init__(self):
        """Initializes application context and not much else."""
        self.data_range = {}
        self.arr = None
        self.app_context = {}

    @property
    def cursor_dims(self) -> List[str]:
        """The dimesnion names for the current cursor order."""
        return self._cursor_dims

    @cursor_dims.setter
    def cursor_dims(self, value):
        self._cursor_dims = value

    @property
    def cursor_dict(self):
        """The location of the cursor in the data volume, as a dim=value dict."""
        if self._cursor_dims is None:
            return None

        return dict(zip(self.cursor_dims, self.cursor))

    @property
    def cursor(self):
        """The location of the cursor in the data volume."""
        return self._cursor

    def add_cursor_lines(self, figure):
        """Adds the standard X and Y cursors to a figure."""
        cursor_lines = figure.multi_line(
            xs=[self._horiz_cursor_x, self._vert_cursor_x],
            ys=[self._horiz_cursor_y, self._vert_cursor_y],
            line_color="white",
            line_width=2,
            line_dash="dotted",
        )
        self._cursor_lines = cursor_lines
        return cursor_lines

    @cursor.setter
    def cursor(self, values):
        from bokeh.models.widgets.markups import Div

        if self._cursor_dims is None:
            try:
                self._cursor_dims = list(self.arr.dims)
            except AttributeError:
                pass

        self._cursor = values
        if self._cursor_info is None:
            self._cursor_info = Div(text="")
            self._horiz_cursor_x = list(self.data_range["x"])
            self._horiz_cursor_y = [0, 0]
            self._vert_cursor_x = [0, 0]
            self._vert_cursor_y = list(self.data_range["y"])
        else:
            self._vert_cursor_x[0] = self._vert_cursor_x[1] = self.cursor[0]
            self._horiz_cursor_y[0] = self._horiz_cursor_y[1] = self.cursor[1]

        self._cursor_info.text = "<h2>Cursor:</h2><span>({})</span>".format(
            ", ".join("{0:.3f}".format(c) for c in self.cursor)
        )

        if self._cursor_lines is not None:
            self._cursor_lines.data_source.data = {
                "xs": [self._horiz_cursor_x, self._vert_cursor_x],
                "ys": [self._horiz_cursor_y, self._vert_cursor_y],
            }

        try:
            self.app_context["cursor_dict"] = dict(zip(self._cursor_dims, self.cursor))
            # self.app_context['full_cursor'] =
        except AttributeError:
            pass


class BokehInteractiveTool(ABC):
    """Base class for Bokeh based analysis applications.

    You should view this as deprecated in light of newer variants based on Qt
    which are much much more performant. There is no way of doing array memory sharing
    with Jupyter so high performance browser based applications for PyARPES are probably
    not going to happen anytime soon.
    """

    auto_rebin = True
    auto_zero_nans = True
    rebin_size = 800

    _debug_div = None

    @property
    def debug_div(self):
        """A debug element which makes developing tools with Bokeh more straightforward."""
        from bokeh.models.widgets.markups import Div

        if self._debug_div is None:
            self._debug_div = Div(text="", width=300, height=100)

        return self._debug_div

    def __setattr__(self, name, value):
        """Overrides __setattr__ to handle setting debug info into a div so it's visible."""
        if name == "debug_text":
            self._debug_div.text = value

        super().__setattr__(name, value)

    def update_colormap_for(self, plot_name):
        """Sets the colormap on the plot `plot_name` to have an appropriate range."""

        def update_plot_colormap(attr, old, new):
            plot_data = self.plots[plot_name].data_source.data["image"]
            low, high = np.min(plot_data), np.max(plot_data)
            dynamic_range = high - low
            self.color_maps[plot_name].update(
                low=low + new[0] / 100 * dynamic_range, high=low + new[1] / 100 * dynamic_range
            )

        return update_plot_colormap

    def init_bokeh_server(self):
        """Tells Bokeh to send output to Jupyter with a relatively long timeout.

        The long timeout is allows for slowish tool startup and transport of large arrays
        over HTTP to the running JS application.
        """
        from bokeh.io import output_notebook

        if "bokeh_configured" not in arpes.config.CONFIG:
            arpes.config.CONFIG["bokeh_configured"] = True

            # use a longer load_timeout for heavy tools
            output_notebook(hide_banner=True, load_timeout=10000)

            # Don't need to manually start a server in the manner of
            # https://matthewrocklin.com/blog//work/2017/06/28/simple-bokeh-server
            # according to
            # https://github.com/bokeh/bokeh/blob/0.12.10/examples/howto/server_embed/notebook_embed.ipynb

    def load_settings(self, **kwargs):
        """Loads a user's settings for interactive tools into the tool.

        Various settings, like the sizes of widgets and panels can be set in user
        settings overrides, and are read here.
        """
        self.settings = arpes.config.SETTINGS.get("interactive", {}).copy()
        for k, v in kwargs.items():
            if k not in self.settings:
                self.settings[k] = v

    @property
    def default_palette(self):
        """Resolves user settings for a color palette to an actual matplotlib palette."""
        from bokeh import palettes

        palette_options = {
            "viridis": palettes.viridis(256),
            "magma": palettes.magma(256),
            "coolwarm": cc.coolwarm,
        }

        return palette_options[self.settings.get("palette", "viridis")]

    def __init__(self):
        """Sets the initial context and initializes the Bokeh server."""
        self.settings = None
        self.app_context = {
            "data": None,
            "plots": {},
            "figures": {},
            "color_maps": {},
            "widgets": {},
        }

        self.init_bokeh_server()

    def __getattribute__(self, item):
        """Allow more convenient use of attributes from self.app_context. This is a bit strange.

        Args:
            item

        Returns:
            The resolved attribute if it is found in `self.app_context`.
        """
        try:
            return super().__getattribute__(item)
        except AttributeError:
            if item in self.app_context:
                return self.app_context[item]

    @abstractmethod
    def tool_handler(self, doc):
        """Hook for the application configuration and widget definition, without boilerplate."""
        pass

    def make_tool(
        self, arr: Union[xr.DataArray, str], notebook_url=None, notebook_handle=True, **kwargs
    ):
        """Starts the Bokeh application in accordance with the Bokeh app docs.

        Attempts to just guess the correct URL for Jupyter which is very error prone.
        """
        from bokeh.application import Application
        from bokeh.application.handlers import FunctionHandler
        from bokeh.io import show

        def generate_url(port):
            if port is None:
                return "localhost:8888"

            return "localhost:{}".format(port)

        if notebook_url is None:
            if "PORT" in arpes.config.CONFIG:
                notebook_url = "localhost:{}".format(arpes.config.CONFIG["PORT"])
            else:
                notebook_url = "localhost:8888"

        if isinstance(arr, str):
            arr = load_data(arr)
            if "cycle" in arr.dims and len(arr.dims) > 3:
                warnings.warn("Summing over cycle")
                arr = arr.sum("cycle", keep_attrs=True)

        if self.auto_zero_nans and {"kx", "ky", "kz", "kp"}.intersection(set(arr.dims)):
            # We need to copy and make sure to clear any nan values, because bokeh
            # does not send these over the wire for some reason
            arr = arr.copy()
            np.nan_to_num(arr.values, copy=False)

        # rebin any axes that have more than 800 pixels
        if self.auto_rebin and np.any(np.asarray(arr.shape) > self.rebin_size):
            reduction = {d: (s // self.rebin_size) + 1 for d, s in arr.S.dshape.items()}
            warnings.warn("Rebinning with {}".format(reduction))

            arr = rebin(arr, reduction=reduction)

            # TODO pass in a reference to the original copy of the array and make sure that
            # preparation tasks move over transparently

        self.arr = arr
        handler = FunctionHandler(self.tool_handler)
        app = Application(handler)
        show(app, notebook_url=notebook_url, notebook_handle=notebook_handle)

        return self.app_context


class SaveableTool(BokehInteractiveTool):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self._last_save = None

    def make_tool(
        self, arr: Union[xr.DataArray, str], notebook_url=None, notebook_handle=True, **kwargs
    ):
        super().make_tool(arr, notebook_url=notebook_url, notebook_handle=notebook_handle, **kwargs)
        return self.app_context

    @property
    def filename(self):
        if self.name is None:
            return None

        return os.path.join(os.getcwd(), "tools", "tool-{}.json".format(self.name))

    @property
    def path(self):
        return None if self.filename is None else Path(self.filename)

    def load_app(self):
        if self.name is None:
            return

        if not self.path.exists():
            return {}

        self.path.parent.mkdir(exist_ok=True)
        with open(self.filename, "r") as f:
            self.deserialize(json.load(f))

    def save_app(self):
        if self.name is None:
            return

        data = self.serialize()
        if deep_equals(data, self._last_save):
            return

        self.path.parent.mkdir(exist_ok=True)
        with open(self.filename, "w") as f:
            self._last_save = data
            json.dump(data, f)

    def deserialize(self, json_data):
        pass

    def serialize(self):
        return {}
