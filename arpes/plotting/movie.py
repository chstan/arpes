import xarray as xr
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation

from plotting.utils import path_for_plot
from provenance import save_plot_provenance

__all__ = ('plot_movie',)


@save_plot_provenance
def plot_movie(data: xr.DataArray, time_dim, interval=None,
               fig=None, ax=None, out=None, **kwargs):
    if not isinstance(data, xr.DataArray):
        raise TypeError('You must provide a DataArray')

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    plot = data.mean(time_dim).plot(vmax=data.max().item(), vmin=data.min().item())

    def init():
        plot.set_array(np.asarray([]))
        return plot,

    animation_coords = data.coords[time_dim].values

    def animate(i):
        coordinate = animation_coords[i]
        data_for_plot = data.sel(**dict([[time_dim, coordinate]]))
        plot.set_array(data_for_plot.values.ravel())
        return plot,

    if interval:
        computed_interval = interval
    else:
        computed_interval = 100

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        repeat=500,
        frames=len(animation_coords), interval=computed_interval, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1000 / computed_interval, metadata=dict(artist='Me'), bitrate=1800)

    if out is not None:
        anim.save(path_for_plot(out), writer=writer)
        return path_for_plot(out)

    plt.show()
    return anim
