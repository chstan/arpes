import matplotlib.pyplot as plt
import itertools

from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.plotting.utils import path_for_plot
from arpes.io import simple_load
from arpes.plotting import annotate_point

__all__ = ('reference_scan_spatial',)


@save_plot_provenance
def reference_scan_spatial(data, out=None, **kwargs):
    data = normalize_to_spectrum(data)

    dims = [d for d in data.dims if d in {'cycle', 'phi', 'eV'}]

    summed_data = data.sum(dims, keep_attrs=True)

    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    flat_axes = list(itertools.chain(*ax))

    summed_data.plot(ax=flat_axes[0])
    flat_axes[0].set_title(r'Full \textbf{eV} range')

    dims_except_eV = [d for d in dims if d != 'eV']
    summed_data = data.sum(dims_except_eV)

    mul = 0.2
    rng = data.coords['eV'].max().item() - data.coords['eV'].min().item()
    offset = data.coords['eV'].max().item()
    if offset > 0:
        offset = 0

    if rng > 3:
        mul = rng / 5.

    for i in range(5):
        low_e, high_e = -mul * (i + 1) + offset, -mul * i + offset
        title = r'\textbf{eV}' + ': {:.2g} to {:.2g}'.format(low_e, high_e)
        summed_data.sel(eV=slice(low_e, high_e)).sum('eV').plot(ax=flat_axes[i+1])
        flat_axes[i + 1].set_title(title)

    y_range = flat_axes[0].get_ylim()
    x_range = flat_axes[0].get_xlim()
    delta_one_percent = ((x_range[1] - x_range[0]) / 100, (y_range[1] - y_range[0]) / 100)

    smart_delta = (2 * delta_one_percent[0], -1.5 * delta_one_percent[0])

    referenced = data.S.referenced_scans

    # idea here is to collect points by those that are close together, then
    # only plot one annotation
    condensed = []
    cutoff = 3 # 3 percent
    for index, row in referenced.iterrows():
        ff = simple_load(index)

        x, y, _ = ff.S.sample_pos
        found = False
        for cx, cy, cl in condensed:
            if abs(cx  - x) < cutoff * abs(delta_one_percent[0]) and abs(cy - y) < cutoff * abs(delta_one_percent[1]):
                cl.append(index)
                found = True
                break

        if not found:
            condensed.append((x, y, [index]))

    for fax in flat_axes:
        for cx, cy, cl in condensed:
            annotate_point(fax, (cx, cy,), ','.join([str(l) for l in cl]),
                           delta=smart_delta, fontsize='large')

    plt.tight_layout()

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax