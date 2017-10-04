import uuid
import warnings

import xarray as xr
import datetime


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
