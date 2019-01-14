import xarray as xr

import matplotlib.pyplot as plt
from arpes.provenance import save_plot_provenance

__all__ = ('plot_parameter',)


@save_plot_provenance
def plot_parameter(fit_data: xr.DataArray, param_name: str, ax=None,
                   fillstyle='none',
                   shift=0, x_shift=0,
                   markersize=8, title=None, out=None, two_sigma=False, **kwargs):
    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7, 5)))

    ds = fit_data.F.param_as_dataset(param_name)
    x = ds.coords[ds.value.dims[0]].values
    print(kwargs)
    color = kwargs.get('color')
    e_width = None
    l_width = None
    if two_sigma:
        line, markers, lines = ax.errorbar(x + x_shift, ds.value.values + shift, yerr=2 * ds.error.values, fmt='', elinewidth=1, linewidth=0, c=color)
        color = lines[0].get_color()[0]
        e_width = 2
        l_width = 0

    ax.errorbar(x + x_shift, ds.value.values + shift, yerr=ds.error.values, fmt='s',
                color=color, elinewidth=e_width, linewidth=l_width,
                markeredgewidth=e_width or 2,
                fillstyle=fillstyle,
                markersize=markersize)