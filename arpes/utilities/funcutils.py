from collections import defaultdict
import functools
import xarray as xr
import time

from arpes.typing import DataType

__all__ = ['Debounce', 'lift_dataarray_to_generic', 'iter_leaves']


def collect_leaves(tree, is_leaf=None):
    """
    Produces a flat representation of the leaves, leaves with the same key are
    collected into a list in the order of appearance, but this depends on the
    dictionary iteration order.

    Example:
    collect_leaves({'a': 1, 'b': 2, 'c': {'a': 3, 'b': 4}}) -> {'a': [1, 3], 'b': [2, 4]}

    :param tree:
    :param is_leaf:
    :return:
    """
    def reducer(dd, item):
        dd[item[0]].append(item[1])
        return dd

    return functools.reduce(reducer, iter_leaves(tree, is_leaf), defaultdict(list))


def iter_leaves(tree, is_leaf=None):
    """
    Iterates across the leaves of a nested dictionary. Whether a particular piece
    of data counts as a leaf is controlled by the predicate `is_leaf`. By default,
    all nested dictionaries are considered not leaves, i.e. an item is a leaf if and
    only if it is not a dictionary.

    Iterated items are returned as key value pairs.

    As an example, you can easily flatten a nested structure with
    `dict(leaves(data))`

    :param tree:
    :param is_leaf:
    :return:
    """
    if is_leaf is None:
        is_leaf = lambda x: not isinstance(x, dict)

    for k, v in tree.items():
        if is_leaf(v):
            yield k, v
        else:
            for item in iter_leaves(v):
                yield item


def lift_dataarray_to_generic(f):
    """
    A functorial decorator that lifts functions with the signature

    (xr.DataArray, *args, **kwargs) -> xr.DataArray

    to one with signature

    A = typing.Union[xr.DataArray, xr.Dataset]
    (A, *args, **kwargs) -> A

    i.e. one that will operate either over xr.DataArrays or xr.Datasets.
    :param f:
    :return:
    """
    @functools.wraps(f)
    def func_wrapper(data: DataType, *args, **kwargs):
        if isinstance(data, xr.DataArray):
            return f(data, *args, **kwargs)
        else:
            assert(isinstance(data, xr.Dataset))
            new_vars = {
                datavar: f(data[datavar], *args, **kwargs) for datavar in data.data_vars
            }

            for var_name, var in new_vars.items():
                if isinstance(var, xr.DataArray) and var.name is None:
                    var.name = var_name

            merged = xr.merge(new_vars.values())
            return merged.assign_attrs(data.attrs)

    return func_wrapper


class Debounce(object):
    def __init__(self, period):
        self.period = period  # never call the wrapped function more often than this (in seconds)
        self.count = 0  # how many times have we successfully called the function
        self.count_rejected = 0  # how many times have we rejected the call
        self.last = None  # the last time it was called

    # force a reset of the timer, aka the next call will always work
    def reset(self):
        self.last = None

    def __call__(self, f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            willcall = False
            if self.last is not None:
                # amount of time since last call
                delta = now - self.last
                if delta >= self.period:
                    willcall = True
                else:
                    willcall = False
            else:
                willcall = True  # function has never been called before

            if willcall:
                # set these first incase we throw an exception
                self.last = now  # don't use time.time()
                self.count += 1
                f(*args, **kwargs)  # call wrapped function
            else:
                self.count_rejected += 1
        return wrapped