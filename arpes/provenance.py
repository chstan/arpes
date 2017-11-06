import datetime
import functools
import json
import os.path
import uuid
import warnings

import xarray as xr

import arpes.config


def attach_id(data):
    if 'id' not in data.attrs:
        data.attrs['id'] = str(uuid.uuid1())


def provenance_from_file(child_arr: xr.DataArray, file, record):
    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    child_arr.attrs['provenance'] = {
        'record': record,
        'file': file,
        'parents_provenance': 'filesystem',
        'time': str(datetime.datetime.now()),
    }


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
            assert (arpes.config.CONFIG['WORKSPACE'] is not None)
            if arpes.config.CONFIG['WORKSPACE'] not in path:
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


def provenance(child_arr: xr.DataArray, parent_arr: xr.DataArray, record, keep_parent_ref=False):
    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    if child_arr.attrs['id'] == parent_arr.attrs['id']:
        warnings.warn('Duplicate id for dataset %s' % child_arr.attrs['id'])

    child_arr.attrs['provenance'] = {
        'record': record,
        'parent_id': parent_arr.attrs['id'],
        'parents_provanence': parent_arr.attrs['provenance'],
        'time': str(datetime.datetime.now()),
    }

    if keep_parent_ref:
        child_arr.attrs['provenance']['parent'] = parent_arr


def provenance_multiple_parents(child_arr: xr.DataArray, parents, record, keep_parent_ref=False):
    if 'id' not in child_arr.attrs:
        attach_id(child_arr)

    if child_arr.attrs['id'] in {p.attrs.get('id', None) for p in parents}:
        warnings.warn('Duplicate id for dataset %s' % child_arr.attrs['id'])

    child_arr.attrs['provenance'] = {
        'record': record,
        'parent_id': [p.attrs['id'] for p in parents],
        'parents_provenance': [p.attrs['provenance'] for p in parents],
        'time': str(datetime.datetime.now()),
    }

    if keep_parent_ref:
        child_arr.attrs['provenance']['parent'] = parents


def _get_provenance(arr: xr.DataArray):
    pass

def show_provenance(arr: xr.DataArray):
    frames = []
