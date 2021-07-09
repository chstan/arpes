"""Utilities used in broadcast fitting."""
from typing import Dict, Union
import lmfit
import xarray as xr
import operator
import warnings
import functools
from string import ascii_lowercase


def unwrap_params(params, iter_coordinate):
    """Inspects array-like parameters and extracts the appropriate value to use for the current fit."""

    def transform_or_walk(v):
        if isinstance(v, dict):
            return unwrap_params(v, iter_coordinate)

        if isinstance(v, xr.DataArray):
            return v.sel(**iter_coordinate, method="nearest").item()

        return v

    return {k: transform_or_walk(v) for k, v in params.items()}


def apply_window(data: xr.DataArray, cut_coords: Dict[str, Union[float, slice]], window):
    """Cuts data inside a specified window.

    Because we allow passing an array of windows, we need to first try to find
    the right one for the current fit before application.

    If there's no window, this acts as the identity function on the data.
    """
    cut_data = data.sel(**cut_coords)
    original_cut_data = cut_data

    if isinstance(window, xr.DataArray):
        window_item = window.sel(**cut_coords).item()
        if isinstance(window_item, slice):
            cut_data = cut_data.sel(**dict([[cut_data.dims[0], window_item]]))

    return cut_data, original_cut_data


def _parens_to_nested(items):
    """Turns a flat list with parentheses tokens into a nested list."""
    parens = [
        (
            token,
            idx,
        )
        for idx, token in enumerate(items)
        if isinstance(token, str) and token in "()"
    ]
    if parens:
        first_idx, last_idx = parens[0][1], parens[-1][1]
        if parens[0][0] != "(" or parens[-1][0] != ")":
            raise ValueError("Parentheses do not match!")

        return (
            items[0:first_idx]
            + [_parens_to_nested(items[first_idx + 1 : last_idx])]
            + items[last_idx + 1 :]
        )
    else:
        return items


def reduce_model_with_operators(model):
    """Combine models according to mathematical operators."""
    if isinstance(model, tuple):
        return model[0](prefix="{}_".format(model[1]), nan_policy="omit")

    if isinstance(model, list) and len(model) == 1:
        return reduce_model_with_operators(model[0])

    left, op, right = model[0], model[1], model[2:]
    left, right = reduce_model_with_operators(left), reduce_model_with_operators(right)

    if op == "+":
        return left + right
    elif op == "*":
        return left * right
    elif op == "-":
        return left - right
    elif op == "/":
        return left / right


def compile_model(model, params=None, prefixes=None):
    """Generates an lmfit model instance from specification.

    Takes a model sequence, i.e. a Model class, a list of such classes, or a list
    of such classes with operators and instantiates an appropriate model.
    """
    if params is None:
        params = {}

    prefix_compile = "{}"
    if prefixes is None:
        prefixes = ascii_lowercase
        prefix_compile = "{}_"

    try:
        if issubclass(model, lmfit.Model):
            return model()
    except TypeError:
        pass

    if isinstance(model, (list, tuple)) and all([isinstance(token, type) for token in model]):
        models = [
            m(prefix=prefix_compile.format(prefixes[i]), nan_policy="omit")
            for i, m in enumerate(model)
        ]
        if isinstance(params, (list, tuple)):
            for cs, m in zip(params, models):
                for name, params_for_name in cs.items():
                    m.set_param_hint(name, **params_for_name)

        built = functools.reduce(operator.add, models)
    else:
        warnings.warn("Beware of equal operator precedence.")
        prefix = iter(prefixes)
        model = [m if isinstance(m, str) else (m, next(prefix)) for m in model]
        built = reduce_model_with_operators(_parens_to_nested(model))

    return built
