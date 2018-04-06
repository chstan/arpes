import matplotlib.pyplot as plt

from arpes.provenance import save_plot_provenance
from .utils import *


__all__ = ('plot_with_bands',)

@save_plot_provenance
def plot_with_bands(data, bands, title=None, ax=None, norm=None,
                    out=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if title is None:
        title = data.S.label.replace('_', ' ')

    mesh = data.plot(norm=norm, ax=ax, **kwargs)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap('Blues')

    for band in bands:
        plt.scatter(band.center.values, band.coords[band.dims[0]].values)

    if out is not None:
        filename = path_for_plot(out)
        plt.savefig(filename)
        return filename
    else:
        plt.show()
        return ax