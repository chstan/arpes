import warnings

import arpes
import os
import sys
import arpes.config
from arpes.endstations import load_scan
from arpes.utilities import modern_clean_xlsx_dataset, \
    attach_extra_dataset_columns, rename_datavar_standard_attrs, \
    clean_datavar_attribute_names
from arpes.utilities.dataset import walk_datasets
from arpes.io import save_dataset, save_dataset_for_export, dataset_exists

__all__ = ('prepare_raw_files',)


def prepare_raw_files(workspace=None, debug=False, reload=False, file=None, quiet=False, export=False, **kwargs):
    import arpes.xarray_extensions

    stop_debug = False
    if debug == 'first':
        stop_debug = True
        debug = True

    arpes.config.attempt_determine_workspace(workspace)
    assert(isinstance(arpes.config.CONFIG['WORKSPACE'], dict))
    workspace_path = arpes.config.CONFIG['WORKSPACE']['path']

    print('Found: {}'.format(workspace_path))
    files = walk_datasets(use_workspace=True)
    if file:
        print("├{}".format(file))
        files = [os.path.join(workspace_path, file)]

    for dataset_path in files:
        # we used to pass "use_soft_match" here, we are switching to a regex based approach so this might
        # initially cause some problems but should be much better in the end
        ds = modern_clean_xlsx_dataset(dataset_path, with_inferred_cols=False, write=True)

        print('└┐')
        for iter_index, (file, scan) in enumerate(ds.iterrows()):
            print(' ├{}'.format(file))
            scan['file'] = scan.get('path', file)
            if not dataset_exists(scan.get('id')) or reload:
                try:
                    with warnings.catch_warnings():
                        if quiet:
                            warnings.simplefilter('ignore')
                        data = load_scan(dict(scan), **kwargs)
                    data = rename_datavar_standard_attrs(data)
                    data = clean_datavar_attribute_names(data)
                    if export:
                        index = scan['file']
                        try:
                            index = int(index)
                        except ValueError:
                            pass
                        if not isinstance(index, int):
                            index = iter_index + 1
                        save_dataset_for_export(data, index, force=True)
                    else:
                        save_dataset(data, force=True)
                except Exception as e:
                    if debug:
                        if stop_debug:
                            debug = False

                        import pdb
                        pdb.post_mortem(e.__traceback__)
                    else:
                        print('Encountered Error {}. Skipping...'.format(e))

        attach_extra_dataset_columns(dataset_path)