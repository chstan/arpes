import json

import xarray as xr

import arpes.config
import arpes.io


def normalize_data(data):
    if isinstance(data, xr.DataArray):
        assert('id' in data.attrs)
        return data.attrs['id']

    if isinstance(data, xr.Dataset):
        raise TypeError('xarray.Dataset is not supported as a normalizable dataset')

    assert(isinstance(data, str))

    return data


def denormalize_data(data):
    if isinstance(data, str):
        # Try to parse it as a UUID, load or retrieve appropriate dataset
        pass

    return data


def computation_hash(pipeline_name, data, intern_kwargs, *args, **kwargs):
    return json.dumps({
        'pipeline_name': pipeline_name,
        'data': normalize_data(data),
        'args': args,
        'kwargs': {k: v for k, v in kwargs.items() if k in intern_kwargs},
    }, sort_keys=True)


def cache_computation(key, data):
    try:
        if isinstance(data, xr.DataArray):
            # intern the computation
            arpes.io.save_dataset(data)
            data = normalize_data(data)

        return data
    except Exception as e:
        raise(e)


class PipelineRollbackException(Exception):
    pass


def pipeline(pipeline_name=None, intern_kwargs=None):
    if intern_kwargs is None:
        intern_kwargs = set()

    # TODO write simple tests for the pipeline flags to ensure correct caching conditions
    def pipeline_decorator(f):
        def func_wrapper(data, flush=False, force=False, debug=False, verbose=True, *args, **kwargs):
            key = computation_hash(pipeline_name or f.__name__,
                                   data, intern_kwargs, *args, **kwargs)
            if debug:
                print(pipeline_name or f.__name__, key)

            def echo(v):
                if verbose:
                    print('{}: {}'.format(pipeline_name or f.__name__, v))

            with open(arpes.config.PIPELINE_JSON_SHELF, 'r') as fp:
                # Currently we are using JSON because of a bug in the implementation
                # of python's shelf that causes it to hang forever.
                # A better long term solution here is to use a database or KV-store
                records = json.load(fp)

            if key in records and not force:
                if (not arpes.io.is_a_dataset(key) or arpes.io.dataset_exists(key)):
                    value = records[key]
                    if flush:
                        flushed = True
                        # remove the record of the cached computation, and delete the filesystem cache
                        # for the computation
                        del records[key]
                        if arpes.io.dataset_exists(key):
                            arpes.io.delete_dataset(key)

                    with open(arpes.config.PIPELINE_JSON_SHELF, 'w') as fp:
                        json.dump(records, fp)

                    echo(value)
                    return value
                else:
                    del records[key]
                    with open(arpes.config.PIPELINE_JSON_SHELF, 'w') as fp:
                        json.dump(records, fp)
                    raise PipelineRollbackException()
            elif flush:
                return None

            try:
                if isinstance(data, str):
                    data = arpes.io.load_dataset(data)
            except ValueError:
                pass
            finally:
                computed = f(data, *args, **kwargs)
                records[key] = cache_computation(key, computed)
                with open(arpes.config.PIPELINE_JSON_SHELF, 'w') as fp:
                    json.dump(records, fp)
                echo(normalize_data(computed))
                return computed

        return func_wrapper

    return pipeline_decorator


def compose(*pipelines):
    def composed(data, *args, **kwargs):
        max_restarts = len(pipelines)
        while max_restarts:
            data_in_process = data

            try:
                for p in pipelines:
                    data_in_process = p(data_in_process, *args, **kwargs)

                return data_in_process
            except PipelineRollbackException as e:
                if not max_restarts:
                    raise e

                max_restarts -= 1
                continue

    return composed
