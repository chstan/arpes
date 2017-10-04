import json
import shelve

import xarray as xr

import arpes.config
import arpes.io


def normalize_data(data):
    if isinstance(data, xr.DataArray):
        return data.attrs['id']

    if isinstance(data, xr.DataSet):
        raise TypeError('xarray.DataSet is not supported as a normalizable dataset')

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
        'kwargs': kwargs.items(),
    }, sort_keys=True)


def cache_computation(key, data):
    if isinstance(data, xr.DataArray):
        # intern the computation
        arpes.io.save_dataset(data)
        data = normalize_data(data)

    with shelve.open(arpes.config.PIPELINE_SHELF) as db:
        if key in db:
            print("Hash value %s already in pipeline shelf!" % key)

        db[key] = data


def tags(tag_name):
    def tags_decorator(func):
        def func_wrapper(name):
            return "<{0}>{1}</{0}>".format(tag_name, func(name))

        return func_wrapper

    return tags_decorator


@tags("p")
def get_text(name):
    return "Hello " + name


def pipeline(pipeline_name):
    def pipeline_decorator(f):
        def func_wrapper(data, *args, **kwargs):
            key = computation_hash(pipeline_name or f.__name__,
                                   data, *args, **kwargs)
            with shelve.open(arpes.config.PIPELINE_SHELF) as db:
                if key in db:
                    return db[key]

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
        for p in pipelines:
            data = p(data)

        return data

    return composed
