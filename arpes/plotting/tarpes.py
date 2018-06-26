import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from arpes.provenance import save_plot_provenance
import matplotlib.animation as manimation

from .utils import *

__all__ = ('plot_subtraction_reference',)

@save_plot_provenance
def plot_subtraction_reference(data, title=None, out=None, norm=None, **kwargs):
    # first we need to sum over dimensions that we do not need

    warnings.warn('Unfinished plot_subtraction_reference')
    return None

    sum_dimensions = {'cycle',}
    sum_dimensions.intersection_update(set(data.dims))

    summed_data = data.sum(*list(sum_dimensions))

    matplotlib.use("Agg")
    if title is None:
        title = data.S.label.replace('_', ' ')

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title or 'Test title', artist='Matplotlib')

    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    l, = plt.plot([], [], 'k-o')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    x0, y0 = 0, 0

    with writer.saving(fig, "writer_test.mp4", 100):
        for i in range(100):
            x0 += 0.1 * np.random.randn()
            y0 += 0.1 * np.random.randn()
            l.set_data(x0, y0)
            writer.grab_frame()