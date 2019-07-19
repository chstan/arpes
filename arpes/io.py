import json
import os.path
import uuid
import numpy as np
import typing
import pickle
from pathlib import Path

import pandas as pd
import xarray as xr

import arpes.config
from arpes.config import DATASET_CACHE_PATH, DATASET_CACHE_RECORD, CLEAVE_RECORD, CONFIG, WorkspaceManager
from arpes.exceptions import ConfigurationError
from arpes.typing import DataType
from arpes.utilities import (wrap_datavar_attrs, unwrap_attrs_dict, unwrap_datavar_attrs,
                             WHITELIST_KEYS, FREEZE_PROPS, clean_xlsx_dataset)
from arpes.endstations import load_scan

__all__ = (
    'simple_load', 'direct_load', 'fallback_load', 'load_dataset', 'save_dataset', 'delete_dataset',
    'load_without_dataset', 'load_example_data',
    'save_dataset_for_export',
    'dataset_exists', 'is_a_dataset', 'load_dataset_attrs', 'easy_pickle',
    'sld', 'dld', 'stitch', 'fld',
)


def load_without_dataset(file: typing.Union[str, Path], location=None, **kwargs):
    file = str(Path(file).absolute())

    if location is None:
        raise ValueError('You must provide a location indicating the endstation or instrument used directly when '
                         'loading data without a dataset.')

    return load_scan(dict(file=file, location=location), **kwargs)


def load_example_data():
    file = Path(__file__).parent.parent / 'resources' / 'example_data' / 'main_chamber_cut_0.fits'
    return load_without_dataset(file=file, location='ALG-MC')


def stitch(df_or_list, attr_or_axis, built_axis_name=None, sort=True):
    """
    Stitches together a sequence of scans or a DataFrame in order to provide a unified dataset along a specified axis

    :param df_or_list: list of the files to load
    :param attr_or_axis: coordinate or attribute in order to promote to an index. I.e. if 't_a' is specified,
    we will create a new axis corresponding to the temperature and concatenate the data along this axis
    :return:
    """

    list_of_files = None
    if isinstance(df_or_list, (pd.DataFrame,)):
        list_of_files = list(df_or_list.index)
    else:
        if not isinstance(df_or_list, (list, tuple,)):
            raise TypeError('Expected an interable for a list of the scans to stitch together')

        list_of_files = list(df_or_list)

    if built_axis_name is None:
        built_axis_name = attr_or_axis

    if len(list_of_files) == 0:
        raise ValueError('Must supply at least one file to stitch')

    loaded = [f if isinstance(f, (xr.DataArray, xr.Dataset)) else simple_load(f)
              for f in list_of_files]

    for i, f in enumerate(loaded):
        value = None
        if isinstance(attr_or_axis, (list, tuple)):
            value = attr_or_axis[i]
        elif attr_or_axis in f.attrs:
            value = f.attrs[attr_or_axis]
        elif attr_or_axis in f.coords:
            value = f.coords[attr_or_axis]

        f.coords[built_axis_name] = value

    if sort:
        loaded.sort(key=lambda x: x.coords[built_axis_name])

    return xr.concat(loaded, dim=built_axis_name)


def file_for_pickle(name):
    p = Path('picklejar', '{}.pickle'.format(name))
    p.parent.mkdir(exist_ok=True)
    return str(p)


def load_pickle(name):
    with open(file_for_pickle(name), 'rb') as f:
        return pickle.load(f)


def save_pickle(data, name):
    pickle.dump(data, open(file_for_pickle(name), 'wb'))


def easy_pickle(data_or_str, name=None):
    if isinstance(data_or_str, str) or name is None:
        return load_pickle(data_or_str)

    assert (isinstance(name, str))
    save_pickle(data_or_str, name)


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


def delete_dataset(arr_or_uuid):
    if isinstance(arr_or_uuid, xr.DataArray):
        return delete_dataset(arr_or_uuid.attrs['id'])

    assert (isinstance(arr_or_uuid, str))

    fname = _filename_for(arr_or_uuid)
    if os.path.exists(fname):
        os.remove(fname)


def save_dataset(arr: DataType, filename=None, force=False):
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
    import arpes.xarray_extensions

    # TODO human readable caching in addition to FS caching
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    filename = filename or _filename_for(arr)
    if filename is None:
        filename = _filename_for(arr)
        attrs_filename = _filename_for_attrs(arr)
    else:
        attrs_filename = filename + '.attrs.json'

    if 'id' in arr.attrs and arr.attrs['id'] in records:
        if force:
            if os.path.exists(filename):
                os.replace(filename, filename + '.keep')
            if os.path.exists(attrs_filename):
                os.replace(attrs_filename, attrs_filename + '.keep')
        else:
            return

    df = arr.attrs.pop('df', None)
    arr.attrs.pop('', None)  # protect against totally stripped attribute names
    arr = wrap_datavar_attrs(arr, original_data=arr)
    ref_attrs = arr.attrs.pop('ref_attrs', None)

    arr.to_netcdf(filename, engine='netcdf4')
    with open(attrs_filename, 'w') as f:
        json.dump(arr.attrs, f)

    if 'id' in arr.attrs:
        first_write = arr.attrs['id'] not in records
        records[arr.attrs['id']] = {
            'file': filename,
            **{k: v for k, v in arr.attrs.items() if k in WHITELIST_KEYS}
        }
    else:
        first_write = False

    # this was a first write
    if first_write:
        with open(DATASET_CACHE_RECORD, 'w') as cache_record:
            json.dump(records, cache_record, sort_keys=True, indent=2)

    if ref_attrs is not None:
        arr.attrs['ref_attrs'] = ref_attrs

    arr = unwrap_datavar_attrs(arr)
    if df is not None:
        arr.attrs['df'] = df


def save_dataset_for_export(arr: DataType, index, **kwargs):
    filename = Path('./export/{}.nc'.format(index))
    filename.parent.mkdir(exist_ok=True)

    save_dataset(arr, str(filename), **kwargs)


def is_a_dataset(dataset):
    if isinstance(dataset, xr.Dataset) or isinstance(dataset, xr.DataArray):
        return True

    if isinstance(dataset, str):
        try:
            _ = uuid.UUID(dataset)
            return True
        except ValueError:
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
        return unwrap_attrs_dict(attrs)


def simple_load(fragment, df: pd.DataFrame=None, workspace=None, basic_prep=True):
    with WorkspaceManager(workspace):
        if df is None:
            from arpes.utilities import default_dataset  # break circular dependency
            df = default_dataset()

        def resolve_fragment(filename):
            return str(filename).split('_')[-1]

        # find a soft match
        files = df.index

        def strip_left_zeros(value):
            if len(value) == 1:
                return value

            return value.lstrip('0')

        if isinstance(fragment, (int, np.int32, np.int64,)):
            numbers = [int(f) for f in [strip_left_zeros(''.join(c for c in resolve_fragment(f) if c.isdigit()))
                                        for f in files] if len(f)]
            index = numbers.index(fragment)
        else:
            fragment = str(fragment)
            matches = [i for i, f in enumerate(files) if fragment in f]
            if len(matches) == 0:
                raise ValueError('No match found for {}'.format(fragment))
            if len(matches) > 1:
                raise ValueError('Unique match not found for {}. Options are: {}'.format(
                    fragment, [files[i] for i in matches]))
            index = matches[0]

        data = load_dataset(dataset_uuid=df.loc[df.index[index]], df=df)

        if basic_prep:
            if 'cycle' in data.indexes and len(data.coords['cycle']) == 1:
                data = data.sum('cycle', keep_attrs=True)

        return data


def direct_load(fragment, df: pd.DataFrame=None, workspace=None, file=None, basic_prep=True, **kwargs):
    """
    Loads a dataset directly, in the same manner that prepare_raw_files does, from the denormalized source format.
    This is useful for testing data loading procedures, and for quickly opening data at beamlines.

    The structure of this is very similar to simple_load, and could be shared. The only differences are in selecting
    the DataFrame with all the files at the beginning, and finally loading the data at the end.
    :param fragment:
    :param df:
    :param file:
    :param basic_prep:
    :return:
    """

    with WorkspaceManager(workspace):
        # first get our hands on a dataframe that has a list of all the files, where to find them on disk, and metadata
        if df is None:
            arpes.config.attempt_determine_workspace(lazy=True)
            if file is None:
                from arpes.utilities import default_dataset  # break circular dependency
                df = default_dataset(with_inferred_cols=False)
            else:
                if not os.path.isabs(file):
                    file = os.path.join(CONFIG['WORKSPACE']['path'], file)

                df = clean_xlsx_dataset(file, with_inferred_cols=False, write=False)

        def resolve_fragment(filename):
            return str(filename).split('_')[-1]

        # find a soft match
        files = df.index

        def strip_left_zeros(value):
            if len(value) == 1:
                return value

            return value.lstrip('0')

        if isinstance(fragment, (int, np.int32, np.int64,)):
            numbers = [int(f) for f in [strip_left_zeros(''.join(c for c in resolve_fragment(f) if c.isdigit()))
                                        for f in files] if len(f)]
            index = numbers.index(fragment)
        else:
            fragment = str(fragment)
            matches = [i for i, f in enumerate(files) if fragment in f]
            if len(matches) == 0:
                raise ValueError('No match found for {}'.format(fragment))
            if len(matches) > 1:
                raise ValueError('Unique match not found for {}. Options are: {}'.format(
                    fragment, [files[i] for i in matches]))
            index = matches[0]

        scan = df.loc[df.index[index]]
        data = load_scan(dict(scan), **kwargs)

        if basic_prep:
            if 'cycle' in data.indexes and len(data.coords['cycle']) == 1:
                data = data.sum('cycle', keep_attrs=True)

        return data


sld = simple_load
dld = direct_load

def fallback_load(*args, **kwargs):
    try:
        return sld(*args, **kwargs)
    except:
        return dld(*args, **kwargs)


fld = fallback_load


def load_dataset(dataset_uuid=None, filename=None, df: pd.DataFrame = None):
    """
    You might want to prefer ``simple_load`` over calling this directly as it is more convenient.

    :param dataset_uuid: UUID of dataset to load, typically you get this from ds.loc['...'].id. This actually also
    accepts a dataframe slice so ds.loc['...'] also works.
    :param df: dataframe to use to lookup the data in. If none is provided, the result of default_dataset is used.
    :return:
    """
    if df is None:
        try:
            from arpes.utilities import default_dataset  # break circular dependency
            df = default_dataset()
        except Exception:
            pass

    if filename is None:
        filename = _filename_for(dataset_uuid)

    if not os.path.exists(filename):
        raise ValueError('%s is not cached on the FS. Did you run `prepare_raw_data`?')

    try:
        arr = xr.open_dataset(filename)
    except ValueError:
        arr = xr.open_dataarray(filename)
    arr = unwrap_datavar_attrs(arr)

    # If the sample is associated with a cleave, attach the information from that cleave
    if 'sample' in arr.attrs and 'cleave' in arr.attrs:
        full_cleave_name = '%s-%s' % (arr.attrs['sample'], arr.attrs['cleave'])

        with open(CLEAVE_RECORD, 'r') as f:
            cleaves = json.load(f)

        skip_keys = {'included_scans', 'note'}
        for k, v in cleaves.get(full_cleave_name, {}).items():
            if k not in skip_keys and k not in arr.attrs:
                arr.attrs[k] = v

    if 'ref_id' in arr.attrs:
        arr.attrs['ref_attrs'] = load_dataset_attrs(arr.attrs['ref_id'])

    for prop in FREEZE_PROPS:
        if prop not in arr.attrs:
            arr.attrs[prop] = getattr(arr.S, prop)

    arr.attrs['df'] = df

    return arr


def available_datasets(**filters):
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    for f, value in filters.items():
        records = {k: v for k, v in records.items() if f in v and v[f] == value}

    return records


def flush_cache(ids, delete=True):
    with open(DATASET_CACHE_RECORD, 'r') as cache_record:
        records = json.load(cache_record)

    for r_id in ids:
        if r_id in records:
            del records[r_id]

        filename = os.path.join(DATASET_CACHE_PATH, r_id + '.nc')
        if os.path.exists(filename) and delete:
            os.remove(filename)

    with open(DATASET_CACHE_RECORD, 'w') as cache_record:
        json.dump(records, cache_record, sort_keys=True, indent=2)
