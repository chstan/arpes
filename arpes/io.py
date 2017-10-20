import json
import os.path
import uuid
import warnings

import xarray as xr

from arpes.config import DATASET_CACHE_PATH, DATASET_CACHE_RECORD, CLEAVE_RECORD


def _filename_for(data):
    if isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        data = data.attrs['id']

    return os.path.join(DATASET_CACHE_PATH, data + '.nc')


_wrappable = {'note', 'data_preparation', 'provenance', 'corrections'}
_ignore_keys = {'Number of slices', 'Time', 'Detector last x-channel', 'Detector first x-channel',
                'Detector first y-channel', 'Detector last y-channel', 'User', 'Step Time', 'ENDSTATION',
                'userPhiOffset', 'MCP', 'provenance', 'Date', 'Version', 'Beam Current', 'userPolarOffset',
                'userNormalIncidenceOffset', 'Energy Step', 'Center Energy'}
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


def save_dataset(arr: xr.DataArray):
    # TODO human readable caching in addition to FS caching
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    if arr.attrs['id'] in records:
        return

    wrap_attrs(arr)
    filename = _filename_for(arr)
    arr.to_netcdf(filename, engine='netcdf4')

    records[arr.attrs['id']] = {
        'file': filename,
        **{k: v for k, v in arr.attrs.items() if k in _whitelist_keys}
    }

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