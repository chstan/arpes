import json
import os.path
import uuid
import warnings

import pandas as pd
import xarray as xr

from arpes.config import DATASET_CACHE_PATH, DATASET_CACHE_RECORD, CLEAVE_RECORD
from arpes.exceptions import ConfigurationError

__all__ = ['load_dataset', 'save_dataset', 'delete_dataset',
           'dataset_exists', 'is_a_dataset', 'load_dataset_attrs']

def _id_for(data):
    if isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        data = data.attrs['id']

    if isinstance(data, pd.Series):
        data = data.id

    if isinstance(data, uuid.UUID):
        data = str(data)

    return data

def _filename_for(data):
    return os.path.join(DATASET_CACHE_PATH, _id_for(data) + '.nc')

def _filename_for_attrs(data):
    return os.path.join(DATASET_CACHE_PATH, _id_for(data) + '.json')


_wrappable = {'note', 'data_preparation', 'provenance', 'corrections', 'symmetry_points'}
_whitelist_keys = {'scan_region', 'sample', 'scan_mode', 'id', 'scan_mode'}


_freeze_props = {'spectrum_type'}

def wrap_attrs(arr: xr.DataArray):
    import arpes.xarray_extensions
    for key in _wrappable:
        if key not in arr.attrs:
            continue

        arr.attrs[key] = json.dumps(arr.attrs[key])

    for prop in _freeze_props:
        if prop not in arr.attrs:
            arr.attrs[prop] = getattr(arr.S, prop)

    if 'time' in arr.attrs:
        arr.attrs['time'] = str(arr.attrs['time'])


def unwrap_attrs_dict(attrs: dict):
    for key in _wrappable:
        if key not in attrs:
            continue

        try:
            attrs[key] = json.loads(attrs[key])
        except json.JSONDecodeError:
            pass


def unwrap_attrs(arr: xr.DataArray):
    for key in _wrappable:
        if key not in arr.attrs:
            continue

        try:
            arr.attrs[key] = json.loads(arr.attrs[key])
        except json.JSONDecodeError:
            pass


def delete_dataset(arr_or_uuid):
    if isinstance(arr_or_uuid, xr.DataArray):
        return delete_dataset(arr_or_uuid.attrs['id'])

    assert(isinstance(arr_or_uuid, str))

    fname = _filename_for(arr_or_uuid)
    if os.path.exists(fname):
        os.remove(fname)


def save_dataset(arr: xr.DataArray, force=False):
    """
    Persists a dataset to disk. In order to serialize some attributes, you may need to modify wrap and unwrap arrs above
    in order to make sure a parameter is saved.

    In some cases, such as when you would like to add information to the attributes,
    it is nice to be able to force a write, since a write would not take place if the file is already on disk.
    To do this you can set the ``force`` attribute.
    :param arr:
    :param force:
    :return:
    """
    # TODO human readable caching in addition to FS caching
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    filename = _filename_for(arr)
    attrs_filename = _filename_for_attrs(arr)
    if arr.attrs['id'] in records:
        if force:
            if os.path.exists(filename):
                os.replace(filename, filename + '.keep')
            if os.path.exists(attrs_filename):
                os.replace(attrs_filename, attrs_filename + '.keep')
        else:
            return

    df = arr.attrs.pop('df', None)
    wrap_attrs(arr)
    ref_attrs = arr.attrs.pop('ref_attrs', None)
    arr.to_netcdf(filename, engine='netcdf4')
    with open(attrs_filename, 'w') as f:
        json.dump(arr.attrs, f)

    first_write = arr.attrs['id'] not in records

    records[arr.attrs['id']] = {
        'file': filename,
        **{k: v for k, v in arr.attrs.items() if k in _whitelist_keys}
    }

    # this was a first write
    if first_write:
        with open(DATASET_CACHE_RECORD, 'w') as cache_record:
            json.dump(records, cache_record, sort_keys=True, indent=2)

    if ref_attrs is not None:
        arr.attrs['ref_attrs'] = ref_attrs

    unwrap_attrs(arr)
    if df is not None:
        arr.attrs['df'] = df


def is_a_dataset(dataset):
    if isinstance(dataset, xr.Dataset) or isinstance(dataset, xr.DataArray):
        return True

    if isinstance(dataset, str):
        try:
            uid = uuid.UUID(dataset)
            return True
        except:
            return False

    return False


def dataset_exists(dataset):
    if isinstance(dataset, xr.Dataset) or isinstance(dataset, xr.DataArray):
        return True

    if isinstance(dataset, str):
        filename = _filename_for(dataset)
        return os.path.exists(filename)

    return False


def load_dataset_attrs(dataset_uuid):
    filename = _filename_for_attrs(dataset_uuid)
    if not os.path.exists(filename):
        try:
            ds = load_dataset(dataset_uuid)
            save_dataset(ds, force=True)
        except ValueError as e:
            raise ConfigurationError(
                'Could not load attributes for {}'.format(dataset_uuid)) from e

    with open(filename, 'r') as f:
        attrs = json.load(f)
        unwrap_attrs_dict(attrs)
        return attrs


def load_dataset(dataset_uuid, df: pd.DataFrame=None):
    if df is None:
        from arpes.utilities import default_dataset # break circular dependency
        df = default_dataset()

    filename = _filename_for(dataset_uuid)
    if not os.path.exists(filename):
        raise ValueError('%s is not cached on the FS')

    arr = xr.open_dataarray(filename)
    unwrap_attrs(arr)

    # If the sample is associated with a cleave, attach the information from that cleave
    if 'sample' in arr.attrs and 'cleave' in arr.attrs:
        full_cleave_name = '%s-%s' % (arr.attrs['sample'], arr.attrs['cleave'])

        with open(CLEAVE_RECORD, 'r') as f:
            cleaves = json.load(f)

        skip_keys = {'included_scans', 'note'}
        for k, v in cleaves.get(full_cleave_name, {}).items():
            if k not in skip_keys and k not in arr.attrs:
                arr.attrs[k] = v
    else:
        warnings.warn('Could not fetch cleave information.')

    if 'ref_id' in arr.attrs:
        arr.attrs['ref_attrs'] = load_dataset_attrs(arr.attrs['ref_id'])

    for prop in _freeze_props:
        if prop not in arr.attrs:
            arr.attrs[prop] = getattr(arr.S, prop)

    arr.attrs['df'] = df

    return arr


def available_datasets(**filters):
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    for filt, value in filters.items():
        records = {k: v for k, v in records.items() if filt in v and v[filt] == value}

    return records

def flush_cache(ids, delete=True):
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    for id in ids:
        if id in records:
            del records[id]

        filename = os.path.join(DATASET_CACHE_PATH, id + '.nc')
        if os.path.exists(filename) and delete:
            os.remove(filename)

    with open(DATASET_CACHE_RECORD, 'w') as cache_record:
        json.dump(records, cache_record, sort_keys=True, indent=2)