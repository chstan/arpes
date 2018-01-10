import warnings

import numpy as np
import xarray as xr
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import show, output_notebook

from abc import ABC, abstractmethod

from arpes.analysis.general import rebin

import arpes.config
from arpes.io import load_dataset
from typing import Union

__all__ = ('BokehInteractiveTool',)

class BokehInteractiveTool(ABC):
    auto_rebin = True
    auto_zero_nans = True
    rebin_size = 800

    def init_bokeh_server(self):
        if 'bokeh_configured' not in arpes.config.CONFIG:
            arpes.config.CONFIG['bokeh_configured'] = True
            output_notebook(hide_banner=True)

            # Don't need to manually start a server in the manner of
            # https://matthewrocklin.com/blog//work/2017/06/28/simple-bokeh-server
            # according to
            # https://github.com/bokeh/bokeh/blob/0.12.10/examples/howto/server_embed/notebook_embed.ipynb

    def __init__(self):
        self.app_context = {
            'data': None,
            'plots': {},
            'figures': {},
            'color_maps': {},
            'widgets': {},
        }

        self.init_bokeh_server()

    @abstractmethod
    def tool_handler(self, doc):
        pass

    def make_tool(self, arr: Union[xr.DataArray, str], notebook_url='localhost:8888',
                  notebook_handle=True, **kwargs):
        if isinstance(arr, str):
            arr = load_dataset(arr)
            if 'cycle' in arr.dims and len(arr.dims) > 3:
                warnings.warn('Summing over cycle')
                arr = arr.sum('cycle', keep_attrs=True)

        if self.auto_zero_nans and len({'kx', 'ky', 'kz', 'kp'}.intersection(set(arr.dims))) > 0:
            # We need to copy and make sure to clear any nan values, because bokeh
            # does not send these over the wire for some reason
            arr = arr.copy()
            np.nan_to_num(arr.values, copy=False)

        # rebin any axes that have more than 800 pixels
        if self.auto_rebin and np.any(np.asarray(arr.shape) > self.rebin_size):
            reduction = {d: (s // self.rebin_size) + 1 for d, s in arr.S.dshape.items()}
            warnings.warn('Rebinning with {}'.format(reduction))

            arr = rebin(arr, reduction=reduction)

            # TODO pass in a reference to the original copy of the array and make sure that
            # preparation tasks move over transparently

        self.arr = arr
        handler = FunctionHandler(self.tool_handler)
        app = Application(handler)
        show(app, notebook_url=notebook_url, notebook_handle=notebook_handle)

        return self.app_context