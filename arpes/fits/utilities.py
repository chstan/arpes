import functools
import warnings
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

import arpes.fits.fit_models
import lmfit


def parse_model(model):
    """
    Takes a model string and turns it into a tokenized version.

    1. ModelClass -> ModelClass
    2. [ModelClass] -> [ModelClass]
    3. str -> [<ModelClass, operator as string>]

    i.e.

    A + (B + C) * D -> [A, '(', B, '+', C, ')', '*', D]

    :param model:
    :return:
    """
    if not isinstance(model, str):
        return model

    pad_all = ['+', '-', '*', '/', '(', ')']

    for pad in pad_all:
        model = model.replace(pad, ' {} '.format(pad))

    special = set(pad_all)

    def read_token(token):
        if token in special:
            return token
        try:
            token = float(token)
            return token
        except ValueError:
            try:
                return arpes.fits.fit_models.__dict__[token]
            except KeyError:
                raise ValueError('Could not find model: {}'.format(token))

    return [read_token(token) for token in model.split()]


def _parens_to_nested(items):
    """
    Turns a flat list with parentheses tokens into a nested list
    :param items:
    :return:
    """

    parens = [(token, idx,) for idx, token in enumerate(items) if isinstance(token, str) and token in '()']
    if len(parens):
        first_idx, last_idx = parens[0][1], parens[-1][1]
        if parens[0][0] != '(' or parens[-1][0] != ')':
            raise ValueError('Parentheses do not match!')

        return items[0:first_idx] + [_parens_to_nested(items[first_idx + 1:last_idx])] + items[last_idx + 1:]
    else:
        return items


def reduce_model_with_operators(model):
    if isinstance(model, tuple):
        return model[0](prefix='{}_'.format(model[1]), nan_policy='omit')

    if isinstance(model, list) and len(model) == 1:
        return reduce_model_with_operators(model[0])

    left, op, right = model[0], model[1], model[2:]
    left, right = reduce_model_with_operators(left), reduce_model_with_operators(right)

    if op == '+':
        return left + right
    elif op == '*':
        return left * right
    elif op == '-':
        return left - right
    elif op == '/':
        return left / right


def compile_model(model, params=None, prefixes=None):
    """
    Takes a model sequence, i.e. a Model class, a list of such classes, or a list
    of such classes with operators and instantiates an appropriate model.
    :param model:
    :return:
    """
    if params is None:
        params = {}

    prefix_compile = '{}'
    if prefixes is None:
        prefixes = ascii_lowercase
        prefix_compile = '{}_'

    try:
        if issubclass(model, lmfit.Model):
            return model()
    except TypeError:
        pass

    if isinstance(model, (list, tuple)) and all([isinstance(token, type) for token in model]):
        models = [m(prefix=prefix_compile.format(prefixes[i]), nan_policy='omit') for i, m in enumerate(model)]
        if isinstance(params, (list, tuple)):
            for cs, m in zip(params, models):
                for name, params_for_name in cs.items():
                    m.set_param_hint(name, **params_for_name)

        built = functools.reduce(operator.add, models)
    else:
        warnings.warn('Beware of equal operator precedence.')
        prefix = iter(prefixes)
        model = [m if isinstance(m, str) else (m, next(prefix)) for m in model]
        built = reduce_model_with_operators(_parens_to_nested(model))

    return built


def unwrap_params(params, iter_coordinate):
    """
    Inspects parameters to see if any are array like and extracts the appropriate value to use for the current
    iteration point.
    :param params:
    :param iter_coordinate:
    :return:
    """
    def transform_or_walk(v):
        if isinstance(v, dict):
            return unwrap_params(v, iter_coordinate)

        if isinstance(v, xr.DataArray):
            return v.sel(**iter_coordinate, method='nearest').item()

        return v

    return {k: transform_or_walk(v) for k, v in params.items()}


def broadcast_model(model_cls: typing.Union[type, TypeIterable],
                    data: DataType, broadcast_dims, params=None, progress=True, dataset=True,
                    weights=None, safe=False, prefixes=None):
    """
    Perform a fit across a number of dimensions. Allows composite models.
    :param model_cls:
    :param data:
    :param broadcast_dims:
    :param params:
    :param progress:
    :param dataset:
    :param weights:
    :param safe:
    :return:
    """
    if params is None:
        params = {}

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

    model = compile_model(parse_model(model_cls), params=params, prefixes=prefixes)
    if isinstance(params, (list, tuple)):
        params = {}

    new_params = model.make_params()

    n_fits = np.prod(np.array(list(template.S.dshape.values())))
    wrap_progress = lambda x, *args, **kwargs: x
    if progress:
        wrap_progress = lambda x, *args, **kwargs: tqdm_notebook(x, *args, **kwargs)

    for indices, cut_coords in wrap_progress(template.T.enumerate_iter_coords(), desc='Fitting', total=n_fits):
        current_params = unwrap_params(params, cut_coords)

        cut_data = data.sel(**cut_coords)
        if safe:
            cut_data = cut_data.T.drop_nan()

        weights_for = None
        if weights is not None:
            weights_for = weights.sel(**cut_coords)

        try:
            fit_result = model.guess_fit(cut_data, params=current_params, weights=weights_for)
        except ValueError:
            fit_result = None

        template.loc[cut_coords] = fit_result

        try:
            residual.loc[cut_coords] = fit_result.residual if fit_result is not None else None
        except ValueError as e:
            if not safe:
                raise e

    if dataset:
        return xr.Dataset({
            'results': template,
            'data': data,
            'residual': residual,
            'norm_residual': residual / data,
        }, residual.coords)

    template.attrs['original_data'] = data
    return template
