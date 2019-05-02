import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from arpes.plotting.utils import imshow_arr, path_for_plot
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

@save_plot_provenance
def false_color_plot(data_r: xr.Dataset, data_g: xr.Dataset, data_b: xr.Dataset, ax=None, out=None, invert=False, pmin=0, pmax=1, **kwargs):
    data_r, data_g, data_b = [normalize_to_spectrum(d) for d in (data_r, data_g, data_b)]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7, 5,)))

    def normalize_channel(channel):
        channel -= np.percentile(channel, 100 * pmin)
        channel[channel > np.percentile(channel, 100 * pmax)] = np.percentile(channel, 100 * pmax)
        channel = channel / np.max(channel)
        return channel

    cs = dict(data_r.coords)
    cs['dim_color'] = [1, 2, 3]

    arr = xr.DataArray(
        np.stack([normalize_channel(data_r.values),
                  normalize_channel(data_g.values),
                  normalize_channel(data_b.values)], axis=-1),
        coords=cs,
        dims=list(data_r.dims) + ['dim_color'],
    )

    if invert:
        vs = arr.values
        vs[vs > 1] = 1
        hsv = matplotlib.colors.rgb_to_hsv(vs)
        hsv[:,:,2] = 1 - hsv[:,:,2]
        arr.values = matplotlib.colors.hsv_to_rgb(hsv)

    imshow_arr(arr, ax=ax)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax

