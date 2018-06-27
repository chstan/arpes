import arpes
import os
import arpes.config
from arpes.models.spectrum import load_scan
from arpes.utilities import modern_clean_xlsx_dataset, \
    attach_extra_dataset_columns, rename_datavar_standard_attrs, \
    clean_datavar_attribute_names
from arpes.utilities.dataset import walk_datasets
from arpes.io import save_dataset, dataset_exists

__all__ = ('prepare_raw_files',)


def prepare_raw_files(workspace=None, reload=False, file=None, **kwargs):
    import arpes.xarray_extensions

    arpes.config.attempt_determine_workspace(workspace)
    assert(isinstance(arpes.config.CONFIG['WORKSPACE'], dict))
    workspace_path = arpes.config.CONFIG['WORKSPACE']['path']

    print('Found: {}'.format(workspace_path))
    files = walk_datasets(use_workspace=True)
    if file:
        print("├{}".format(file))
        files = [os.path.join(workspace_path, file)]

    for dataset_path in files:
        ds = modern_clean_xlsx_dataset(dataset_path, with_inferred_cols=False, write=True, allow_soft_match=True)

        print('└┐')
        for file, scan in ds.iterrows():
            print(' ├{}'.format(file))
            scan['file'] = scan.get('path', file)
            if not dataset_exists(scan.get('id')) or reload:
                try:
                    data = load_scan(dict(scan), **kwargs)
                    data = rename_datavar_standard_attrs(data)
                    data = clean_datavar_attribute_names(data)
                    save_dataset(data, force=True)
                except Exception as e:
                    print('Encountered Error {}. Skipping...'.format(e))

        attach_extra_dataset_columns(dataset_path)