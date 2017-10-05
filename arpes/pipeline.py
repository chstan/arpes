import json
import shelve

import xarray as xr

import arpes.config
import arpes.io


def normalize_data(data):
    if isinstance(data, xr.DataArray):
        return data.attrs['id']

    if isinstance(data, xr.Dataset):
        raise TypeError('xarray.Dataset is not supported as a normalizable dataset')

    return data


def denormalize_data(data):
    if isinstance(data, str):
        # Try to parse it as a UUID, load or retrieve appropriate dataset
        pass

    return data


def computation_hash(pipeline_name, data, *args, **kwargs):
    return json.dumps({
        'pipeline_name': pipeline_name,
        'data': normalize_data(data),
        'args': args,
        'kwargs': list(kwargs.items()),
    }, sort_keys=True)


def cache_computation(key, data):
    try:
        if isinstance(data, xr.DataArray):
            # intern the computation
            arpes.io.save_dataset(data)
            data = normalize_data(data)
        with shelve.open(arpes.config.PIPELINE_SHELF) as db:
            if key in db:
                print("Hash value %s already in pipeline shelf!" % key)

            db[key] = data
    except Exception as e:
        raise(e)


class PipelineRollbackException(Exception):
    pass


def pipeline(pipeline_name=None):
    def pipeline_decorator(f):
        def func_wrapper(data, *args, **kwargs):
            key = computation_hash(pipeline_name or f.__name__,
                                   data, *args, **kwargs)
            with shelve.open(arpes.config.PIPELINE_SHELF) as db:
                if key in db:
                    if (not arpes.io.is_a_dataset(key) or arpes.io.dataset_exists(key)):
                        return db[key]
                    else:
                        del db[key]
                        raise PipelineRollbackException()

                try:
                    if isinstance(data, str):
                        data = arpes.io.load_dataset(data)
                except ValueError:
                    pass
                finally:
                    computed = f(data, *args, **kwargs)
                    cache_computation(key, computed)
                    return computed

        return func_wrapper

    return pipeline_decorator


def compose(*pipelines):
    def composed(data):
        max_restarts = len(pipelines)
        while max_restarts:
            data_in_process = data

            try:
                for p in pipelines:
                    data_in_process = p(data_in_process)

                return data_in_process
            except PipelineRollbackException as e:
                if not max_restarts:
                    raise(e)

                max_restarts -= 1
                continue

    return composed
