import datetime
import functools
import json
import os.path
import uuid
import warnings

import xarray as xr

import arpes.config
import typing
from arpes.typing import *


def attach_id(data):
    if 'id' not in data.attrs:
        data.attrs['id'] = str(uuid.uuid1())


def provenance_from_file(child_arr: typing.Union[xr.DataArray, xr.Dataset], file, record):
    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    child_arr.attrs['provenance'] = {
        'record': record,
        'file': file,
        'parents_provenance': 'filesystem',
        'time': datetime.datetime.now().isoformat(),
    }


def update_provenance(what, record_args=None, keep_parent_ref=False):
    """
    Provides a decorator that promotes a function to one that records data provenance.

    :param what: Description of what transpired, to put into the record.
    :param record_args: Unused presently, will allow recording args into record.
    :param keep_parent_ref: Whether to keep a pointer to the parents in the hierarchy or not.
    :return: decorator
    """
    def update_provenance_decorator(f):
        @functools.wraps(f)
        def func_wrapper(*args, **kwargs):

            arg_parents = [v for v in args if isinstance(v, xr_types) and 'id' in v.attrs]
            kwarg_parents = {k: v for k, v in kwargs.items()
                             if isinstance(v, xr_types) and 'id' in v.attrs}
            all_parents = arg_parents + list(kwarg_parents.values())
            result = f(*args, **kwargs)

            # we do not want to record provenance or change the id if ``f`` opted not to do anything
            # to its input. This reduces the burden on client code by allowing them to return the input
            # without changing the 'id' attr
            result_not_identity = not any(p is result for p in all_parents)

            if isinstance(result, xr_types) and result_not_identity:
                if 'id' in result.attrs:
                    del result.attrs['id']

                provenance_fn = provenance
                if len(all_parents) > 1:
                    provenance_fn = provenance_multiple_parents

                if len(all_parents) > 0:
                    provenance_fn(result, all_parents, {
                        'what': what,
                        'by': f.__name__,
                        'time': datetime.datetime.now().isoformat()
                    }, keep_parent_ref)

            return result
        return func_wrapper
    return update_provenance_decorator


def save_plot_provenance(plot_fn):
    """
    A decorator that automates saving the provenance information for a particular plot.
    A plotting function creates an image or movie resource at some location on the
    filesystem.

    In order to hook into this decorator appropriately, because there is no way that I know
    of of temporarily overriding the behavior of the open builtin in order to monitor
    for a write.

    :param plot_fn: A plotting function to decorate
    :return:
    """
    @functools.wraps(plot_fn)
    def func_wrapper(*args, **kwargs):
        path = plot_fn(*args, **kwargs)
        if isinstance(path, str) and os.path.exists(path):
            workspace = arpes.config.CONFIG['WORKSPACE']
            assert(workspace is not None)

            try:
                workspace = workspace['name']
            except TypeError:
                pass

            if workspace not in path:
                warnings.warn(('Plotting function {} appears not to abide by '
                               'practice of placing plots into designated workspaces.').format(plot_fn.__name__))

            provenance_context = {
                'VERSION': arpes.config.CONFIG['VERSION'],
                'time': datetime.datetime.now().isoformat(),
                'name': plot_fn.__name__,
                'args': [arg.attrs.get('provenance', {}) for arg in args
                         if isinstance(arg, xr.DataArray)],
                'kwargs': {k: v.attrs.get('provenance', {}) for k, v in kwargs.items()
                           if isinstance(v, xr.DataArray)}
            }

            provenance_path = path + '.provenance.json'
            with open(provenance_path, 'w') as f:
                json.dump(provenance_context, f, indent=2)

        return path

    return func_wrapper


def provenance(child_arr: xr.DataArray, parent_arr: typing.Union[DataType, typing.List[DataType]], record, keep_parent_ref=False):
    if isinstance(parent_arr, list):
        assert(len(parent_arr) == 1)
        parent_arr = parent_arr[0]

    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    parent_id = parent_arr.attrs.get('id')
    if parent_id is None:
        warnings.warn('Parent array has no ID.')

    if child_arr.attrs['id'] == parent_id:
        warnings.warn('Duplicate id for dataset %s' % child_arr.attrs['id'])

    child_arr.attrs['provenance'] = {
        'record': record,
        'parent_id': parent_id,
        'parents_provanence': parent_arr.attrs.get('provenance'),
        'time': datetime.datetime.now().isoformat(),
    }

    if keep_parent_ref:
        child_arr.attrs['provenance']['parent'] = parent_arr


def provenance_multiple_parents(child_arr: xr_types, parents, record, keep_parent_ref=False):
    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    if child_arr.attrs['id'] in {p.attrs.get('id', None) for p in parents}:
        warnings.warn('Duplicate id for dataset %s' % child_arr.attrs['id'])

    child_arr.attrs['provenance'] = {
        'record': record,
        'parent_id': [p.attrs['id'] for p in parents],
        'parents_provenance': [p.attrs['provenance'] for p in parents],
        'time': datetime.datetime.now().isoformat(),
    }

    if keep_parent_ref:
        child_arr.attrs['provenance']['parent'] = parents
