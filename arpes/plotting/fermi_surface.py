import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.path
import numpy as np

from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.typing import DataType

from arpes.plotting.utils import path_for_plot, path_for_holoviews

__all__ = ('fermi_surface_slices', 'magnify_circular_regions_plot',)

@save_plot_provenance
def fermi_surface_slices(arr: xr.DataArray, n_slices=9, ev_per_slice=0.02, bin=0.01, out=None, **kwargs):
    import holoviews as hv
    slices = []
    for i in range(n_slices):
        high = - ev_per_slice * i
        low = high - bin
        image = hv.Image(arr.sum([d for d in arr.dims if d not in ['theta', 'beta', 'phi', 'eV', 'kp', 'kx', 'ky']]).sel(
            eV=slice(low, high)).sum('eV'), label='%g eV' % high)

        slices.append(image)

    layout = hv.Layout(slices).cols(3)
    if out is not None:
        renderer = hv.renderer('matplotlib').instance(fig='svg', holomap='gif')
        filename = path_for_plot(out)
        renderer.save(layout, path_for_holoviews(filename))
        return filename
    else:
        return layout


@save_plot_provenance
def magnify_circular_regions_plot(data: DataType, magnified_points, mag=10, radius=0.05, cmap='viridis', color=None, edgecolor='red', out=None, ax=None, **kwargs):
    data = normalize_to_spectrum(data)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7, 5,)))

    mesh = data.plot(ax=ax, cmap=cmap)
    clim = list(mesh.get_clim())
    clim[1] = clim[1] / mag

    mask = np.zeros(shape=(len(data.values.ravel()),))
    pts = np.zeros(shape=(len(data.values.ravel()), 2,))
    mask = mask > 0

    raveled = data.T.ravel()
    pts[:,0] = raveled[data.dims[0]]
    pts[:,1] = raveled[data.dims[1]]

    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    width = radius * maxd / dx * (xlim[1] - xlim[0])
    height = radius * maxd / dy * (ylim[1] - ylim[0])

    if not isinstance(edgecolor, list):
        edgecolor = [edgecolor for _ in range(len(magnified_points))]

    if not isinstance(color, list):
        color = [color for _ in range(len(magnified_points))]

    pts[:,1] = (pts[:,1]) / (xlim[1] - xlim[0])
    pts[:,0] = (pts[:,0]) / (ylim[1] - ylim[0])
    print(np.min(pts[:, 1]), np.max(pts[:, 1]))
    print(np.min(pts[:, 0]), np.max(pts[:, 0]))

    for c, ec, point in zip(color, edgecolor, magnified_points):
        patch = matplotlib.patches.Ellipse(point, width, height, color=c, edgecolor=ec, fill=False, linewidth=2, zorder=4)
        patchfake = matplotlib.patches.Ellipse(
            [point[1], point[0]],
            radius, radius
        )
        ax.add_patch(patch)
        mask = np.logical_or(mask, patchfake.contains_points(pts))


    data_masked = data.copy(deep=True)
    data_masked.values = np.array(data_masked.values, dtype=np.float32)

    cm = matplotlib.cm.get_cmap(name='viridis')
    cm.set_bad(color=(1, 1, 1, 0))
    data_masked.values[np.swapaxes(np.logical_not(mask.reshape(data.values.shape[::-1])), 0, 1)] = np.nan

    aspect = ax.get_aspect()
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    ax.imshow(data_masked.values, cmap=cm, extent=extent, zorder=3, clim=clim, origin='lower')
    ax.set_aspect(aspect)

    for spine in ['left', 'top', 'right', 'bottom']:
        ax.spines[spine].set_zorder(5)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
