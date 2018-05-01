import numpy as np
import xarray as xr
from arpes.typing import DataType
from tqdm import tqdm_notebook

from arpes.utilities import normalize_to_spectrum

__all__ = ('broadcast_model',)


def broadcast_model(model_cls: type, data: DataType, broadcast_dims, progress=True, constraints=None, dataset=True):
    if constraints is None:
        constraints = {}

    if isinstance(broadcast_dims, str):
        broadcast_dims = [broadcast_dims]

    data = normalize_to_spectrum(data)
    cs = {}
    for dim in broadcast_dims:
        cs[dim] = data.coords[dim]

    other_axes = set(data.dims).difference(set(broadcast_dims))
    template = data.sum(list(other_axes))
    template.values = np.ndarray(template.shape, dtype=np.object)

    residual = data.copy(deep=True)
    residual.values = np.zeros(residual.shape)

    model = model_cls()

    n_fits = np.prod(np.array(list(template.S.dshape.values())))
    wrap_progress = lambda x, *args, **kwargs: x
    if progress:
        wrap_progress = lambda x, *args, **kwargs: tqdm_notebook(x, *args, **kwargs)

    for indices, cut_coords in wrap_progress(template.T.enumerate_iter_coords(), desc='Fitting',
                                             total=n_fits):
        cut_data = data.sel(**cut_coords)
        fit_result = model.guess_fit(cut_data, params=constraints)
        template.loc[cut_coords] = fit_result
        residual.loc[cut_coords] = fit_result.residual

    if dataset:
        return xr.Dataset({
            'results': template,
            'data': data,
            'residual': residual,
            'norm_residual': residual / data,
        }, residual.coords)

    template.attrs['original_data'] = data
    return template
