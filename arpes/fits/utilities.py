import functools
from string import ascii_lowercase
import operator
import numpy as np
import xarray as xr

import typing
from arpes.typing import DataType
from tqdm import tqdm_notebook

from arpes.utilities import normalize_to_spectrum

__all__ = ('broadcast_model',)


TypeIterable = typing.Union[typing.List[type], typing.Tuple[type]]

def broadcast_model(model_cls: typing.Union[type, TypeIterable],
                    data: DataType, broadcast_dims, constraints=None, progress=True, dataset=True,
                    weights=None):
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

    new_params = None
    if isinstance(model_cls, (list, tuple)):
        models = [m(prefix='{}_'.format(ascii_lowercase[i])) for i, m in enumerate(model_cls)]
        if isinstance(constraints, (list, tuple)):
            for cs, m in zip(constraints, models):
                for name, constraints_for_name in cs.items():
                    m.set_param_hint(name, **constraints_for_name)

            constraints = {}

        model = functools.reduce(operator.add, models)
        new_params = model.make_params()
    else:
        model = model_cls()

    n_fits = np.prod(np.array(list(template.S.dshape.values())))
    wrap_progress = lambda x, *args, **kwargs: x
    if progress:
        wrap_progress = lambda x, *args, **kwargs: tqdm_notebook(x, *args, **kwargs)

    for indices, cut_coords in wrap_progress(template.T.enumerate_iter_coords(), desc='Fitting',
                                             total=n_fits):
        cut_data = data.sel(**cut_coords)
        weights_for = None
        if weights is not None:
            weights_for = weights.sel(**cut_coords)
        fit_result = model.guess_fit(cut_data, params=constraints, weights=weights_for)

        template.loc[cut_coords] = fit_result
        residual.loc[cut_coords] = fit_result.residual if fit_result is not None else None

    if dataset:
        return xr.Dataset({
            'results': template,
            'data': data,
            'residual': residual,
            'norm_residual': residual / data,
        }, residual.coords)

    template.attrs['original_data'] = data
    return template
