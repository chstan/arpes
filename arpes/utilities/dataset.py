import os
import uuid
import warnings

import numpy as np
import pandas as pd

import arpes.config
from arpes.exceptions import ConfigurationError

__all__ = ['clean_xlsx_dataset', 'default_dataset', 'infer_data_path']

_DATASET_EXTENSIONS = {'.xlsx', '.xlx',}
_SEARCH_DIRECTORIES = ('', 'hdf5', 'fits',)
_TOLERATED_EXTENSIONS = {'.h5', '.nc', '.fits',}


def is_blank(item):
    if isinstance(item, str):
        return item == ''

    if isinstance(item, float):
        return np.isnan(item)

    if pd.isnull(item):
        return True

    return False


def infer_data_path(file, scan_desc, allow_soft_match=False):
    if not isinstance(file, str):
        file = str(file)

    if 'path' in scan_desc and not is_blank(scan_desc['path']):
        return scan_desc['path']

    assert('WORKSPACE' in arpes.config.CONFIG)

    base_dir = os.path.join(arpes.config.DATA_PATH, arpes.config.CONFIG['WORKSPACE'])
    dir_options = [os.path.join(base_dir, option) for option in _SEARCH_DIRECTORIES]

    for dir in dir_options:
        try:
            files = filter(lambda f: os.path.splitext(f)[1] in _TOLERATED_EXTENSIONS, os.listdir(dir))
            for f in files:
                if os.path.splitext(file)[0] == os.path.splitext(f)[0]:
                    return os.path.join(dir, f)
                if allow_soft_match and file in os.path.splitext(f)[0]:
                    return os.path.join(dir, f) # soft match

        except FileNotFoundError:
            pass

    raise ConfigurationError('Could not find file associated to {}'.format(file))


def default_dataset(**kwargs):
    material_class = arpes.config.CONFIG['WORKSPACE']
    if material_class is None:
        raise ConfigurationError('You need to set the WORKSPACE attribute on CONFIG!')

    dir = os.path.join(arpes.config.SOURCE_PATH, 'datasets', material_class)

    def is_dataset(filename):
        rest, ext = os.path.splitext(filename)
        rest, internal_ext = os.path.splitext(rest)

        return ext in _DATASET_EXTENSIONS and internal_ext != '.cleaned'

    candidates = list(filter(is_dataset, os.listdir(dir)))
    assert(len(candidates) == 1)

    return clean_xlsx_dataset(os.path.join(dir, candidates[0]), **kwargs)


def clean_xlsx_dataset(path, allow_soft_match=False, **kwargs):
    reload = kwargs.pop('reload', False)
    base_filename, extension = os.path.splitext(path)
    if extension not in _DATASET_EXTENSIONS:
        warnings.warn('File is not an excel file')
        return None

    new_filename = base_filename + '.cleaned' + extension
    if os.path.exists(new_filename):
        if reload:
            os.remove(new_filename)
        else:
            return pd.read_excel(new_filename).set_index('file')

    ds = pd.read_excel(path, **kwargs)
    ds = ds.loc[ds.index.dropna()]

    last_index = None

    # Add required columns
    if 'id' not in ds:
        ds['id'] = np.nan

    if 'path' not in ds:
        ds['path'] = ''

    # Cascade blank values
    for index, row in ds.sort_index().iterrows():
        row = row.copy()

        for key, value in row.iteritems():
            if key == 'id' and np.isnan(float(row['id'])):
                ds.loc[index, ('id',)] = str(uuid.uuid1())

            elif key == 'path' and is_blank(value):
                ds.loc[index, ('path',)] = infer_data_path(row['file'], row, allow_soft_match)

            elif last_index is not None and is_blank(value) and not is_blank(ds.loc[last_index,(key,)]):
                ds.loc[index,(key,)] = ds.loc[last_index,(key,)]

        last_index = index

    excel_writer = pd.ExcelWriter(new_filename)
    ds.to_excel(excel_writer)
    excel_writer.save()

    return ds.set_index('file')

