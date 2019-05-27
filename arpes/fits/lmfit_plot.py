"""
Monkeypatch the lmfit plotting to avoid TeX errors, and to allow plotting model results in 2D.
"""
import xarray as xr
import matplotlib.pyplot as plt
from lmfit import model

original_plot = model.ModelResult.plot


def transform_lmfit_titles(l, is_title=False):
    if is_title:
        l = l.replace('_', '-')

    return l or ''


def patched_plot(self, *args, **kwargs):
    from arpes.plotting.utils import transform_labels

    try:
        if self.model.n_dims != 1:
            from arpes.plotting.utils import fancy_labels

            fig, ax = plt.subplots(2,2, figsize=(10,8))

            def to_dr(flat_data):
                shape = [len(self.independent[d]) for d in self.independent_order]
                return xr.DataArray(flat_data.reshape(shape), coords=self.independent, dims=self.independent_order)

            to_dr(self.init_fit).plot(ax=ax[1][0])
            to_dr(self.data).plot(ax=ax[0][1])
            to_dr(self.best_fit).plot(ax=ax[0][0])
            to_dr(self.residual).plot(ax=ax[1][1])

            ax[0][0].set_title('Best fit')
            ax[0][1].set_title('Data')
            ax[1][0].set_title('Initial fit')
            ax[1][1].set_title('Residual (Data - Best fit)')

            for axi in ax.ravel():
                fancy_labels(axi)

            plt.tight_layout()
            return ax

    except AttributeError:
        pass

    ret = original_plot(self, *args, **kwargs)
    transform_labels(transform_lmfit_titles)
    return ret


model.ModelResult.plot = patched_plot
