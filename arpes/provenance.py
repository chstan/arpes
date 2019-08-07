"""
Provides data provenance for PyARPES. Most analysis routines built
into PyARPES support provenance. Of course, Python is a dynamic language and nothing can be
done to prevent the experimenter from circumventing the provenance scheme.

All the same, between analysis notebooks and the data provenenace provided by PyARPES,
we provide an environment with much higher standard for reproducible analysis than many
other current analysis environments.

This provenenace record is automatically exported when using the built in
plotting utilities. Additionally, passing `used_data` to the PyARPES `savefig`
wrapper allows saving provenance information even for bespoke plots created in
a Jupyter cell.

PyARPES also makes it easy to opt into data provenance for new analysis
functions by providing convenient decorators. These decorators inspect data passed at runtime
to look for and update provenance entries on arguments and return values.
"""

import datetime
import functools
import json
import os.path
import typing
import uuid
import warnings

from typing import Any

import xarray as xr

import arpes.config
from arpes import VERSION
from arpes.typing import xr_types, DataType


def attach_id(data: xr.DataArray) -> None:
    """
    Ensures that an ID is attached to a piece of data, if it does not already exist.
    IDs are generated at the time of identification in an analysis notebook. Sometimes a piece of
    data is created from nothing, and we might need to generate one for it on the spot.
    :param data:
    :return:
    """
    if 'id' not in data.attrs:
        data.attrs['id'] = str(uuid.uuid1())


def provenance_from_file(child_arr: typing.Union[xr.DataArray, xr.Dataset], file, record):
    """
    Builds a provenance entry for a dataset corresponding to loading data from a file. This is used
    by data loaders at the start of an analysis.
    :param child_arr:
    :param file:
    :param record:
    :return:
    """

    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    child_arr.attrs['provenance'] = {
        'record': record,
        'file': file,
        'parents_provenance': 'filesystem',
        'time': datetime.datetime.now().isoformat(),
        'version': VERSION,
    }


def update_provenance(what, record_args=None, keep_parent_ref=False):
    """
    Provides a decorator that promotes a function to one that records data provenance.

    :param what: Description of what transpired, to put into the record.
    :param record_args: Unused presently, will allow recording args into record.
    :param keep_parent_ref: Whether to keep a pointer to the parents in the hierarchy or not.
    :return: decorator
    """
    def update_provenance_decorator(fn):
        @functools.wraps(fn)
        def func_wrapper(*args: Any, **kwargs: Any) -> xr.DataArray:

            arg_parents = [v for v in args if isinstance(v, xr_types) and 'id' in v.attrs]
            kwarg_parents = {k: v for k, v in kwargs.items()
                             if isinstance(v, xr_types) and 'id' in v.attrs}
            all_parents = arg_parents + list(kwarg_parents.values())
            result = fn(*args, **kwargs)

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

                if all_parents:
                    provenance_fn(result, all_parents, {
                        'what': what,
                        'by': fn.__name__,
                        'time': datetime.datetime.now().isoformat(),
                        'version': VERSION,
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

            try:
                workspace = workspace['name']
            except (TypeError, KeyError):
                pass

            if not workspace or workspace not in path:
                warnings.warn(('Plotting function {} appears not to abide by '
                               'practice of placing plots into designated workspaces.').format(plot_fn.__name__))

            provenance_context = {
                'VERSION': VERSION,
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
    """
    Function that updates the provenance for a piece of data with a single parent.

    :param child_arr:
    :param parent_arr:
    :param record:
    :param keep_parent_ref:
    :return:
    """
    if isinstance(parent_arr, list):
        assert len(parent_arr) == 1
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
        'version': VERSION,
    }

    if keep_parent_ref:
        child_arr.attrs['provenance']['parent'] = parent_arr


def provenance_multiple_parents(child_arr: xr_types, parents, record, keep_parent_ref=False):
    """
    Similar to `provenance` updates the data provenance information for data with
    multiple sources or "parents". For instance, if you normalize a piece of data "X" by a metal
    reference "Y", then the returned data would list both "X" and "Y" in its history.

    :param child_arr:
    :param parents:
    :param record:
    :param keep_parent_ref:
    :return:
    """
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
