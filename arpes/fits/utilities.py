import numpy as np
import xarray as xr

__all__ = ['broadcast_model']


def broadcast_model(model_cls: type, dataset: xr.DataArray, broadcast_axis):
    cs = {}
    cs[broadcast_axis] = dataset.coords[broadcast_axis]

    model = model_cls()
    broadcast_values = dataset.coords[broadcast_axis].values
    fit_results = [model.guess_fit(dataset.sel(**dict([(broadcast_axis, v,)])))
                   for v in broadcast_values]

    return xr.DataArray(np.asarray(fit_results), coords=cs, dims=[broadcast_axis])
