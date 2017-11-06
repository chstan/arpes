import json
import os.path
import uuid
import warnings

import xarray as xr

from arpes.config import DATASET_CACHE_PATH, DATASET_CACHE_RECORD, CLEAVE_RECORD

__all__ = ['load_dataset', 'save_dataset', 'delete_dataset',
           'dataset_exists', 'is_a_dataset']

def _filename_for(data):
    if isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        data = data.attrs['id']

    return os.path.join(DATASET_CACHE_PATH, data + '.nc')


_wrappable = {'note', 'data_preparation', 'provenance', 'corrections', 'symmetry_points'}
_whitelist_keys = {'scan_region', 'sample', 'scan_mode', 'id', 'scan_mode'}


def wrap_attrs(arr: xr.DataArray):
    for key in _wrappable:
        if key not in arr.attrs:
            continue

        arr.attrs[key] = json.dumps(arr.attrs[key])

    if 'time' in arr.attrs:
        arr.attrs['time'] = str(arr.attrs['time'])


def unwrap_attrs(arr: xr.DataArray):
    for key in _wrappable:
        if key not in arr.attrs:
            continue

        try:
            arr.attrs[key] = json.loads(arr.attrs[key])
        except Exception as e:
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

    fname = _filename_for(arr)
    if arr.attrs['id'] in records:
        if force:
            if os.path.exists(fname):
                os.replace(fname, fname + '.keep')
        else:
            return

    wrap_attrs(arr)
    filename = _filename_for(arr)
    arr.to_netcdf(filename, engine='netcdf4')

    first_write = arr.attrs['id'] not in records

    records[arr.attrs['id']] = {
        'file': filename,
        **{k: v for k, v in arr.attrs.items() if k in _whitelist_keys}
    }

    # this was a first write
    if first_write:
        with open(DATASET_CACHE_RECORD, 'w') as cache_record:
            json.dump(records, cache_record, sort_keys=True, indent=2)

    unwrap_attrs(arr)


def is_a_dataset(dataset):
    if isinstance(dataset, xr.Dataset) or isinstance(dataset, xr.DataArray):
        return True

    if isinstance(dataset, str):
        try:
            uid = uuid.Uuid(dataset)
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


def load_dataset(dataset_uuid):
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