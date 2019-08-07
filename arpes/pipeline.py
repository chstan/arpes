"""
Although it is not preferred, we support building data analysis pipelines.

As an analogy, this is kind of the "dual" in the sense of vector space duality:
rather than starting with data and doing things with it to perform an analysis,
you can specify how to chain and apply operations to the analysis functions and build
a pipeline or sequence of operations that can in the end be applied to data.

This has some distinct advantages:

1. You can cache computations before expensive steps, and restart calculations part way through.
   In fact, this is already supported in PyARPES. For instance, you can specify a pipline that does
       i.   An expensive correction
       ii.  Momentum space conversion
       iii. An expensive analysis in momentum space
   If you run data through this pipeline and have to stop during step ii., the next time
   you run the pipeline, it will start with the cached result from step i. instead of recomputing
   this value.
2. Systematizing certain kinds of analysis

The core of this is `compose` which takes two pipelines (including atomic elements
like single functions) and returns their composition, which can be paused and restarted between
the two parts. Atomic elements can be constructed with `pipeline`.

In practice though, much of ARPES analysis occurs at scales too small to make this useful,
and interactivity tends to be much preferred to rigidity. PyARPES nevertheless offers this as an
option, as well as trying to provide support for reproducible and understandable scientific
analyses without sacrificing interativity and a tight feedback loop for the experimenter.
"""

import json

from typing import Union

import xarray as xr

import arpes.config
import arpes.io


def normalize_data(data: Union[xr.DataArray, xr.Dataset, str]):
    if isinstance(data, xr.DataArray):
        assert 'id' in data.attrs
        return data.attrs['id']

    if isinstance(data, xr.Dataset):
        raise TypeError('xarray.Dataset is not supported as a normalizable dataset')

    assert isinstance(data, str)

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
        raise e


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

            with open(arpes.config.PIPELINE_JSON_SHELF, 'r') as file:
                # Currently we are using JSON because of a bug in the implementation
                # of python's shelf that causes it to hang forever.
                # A better long term solution here is to use a database or KV-store
                records = json.load(file)

            if key in records and not force:
                if (not arpes.io.is_a_dataset(key) or arpes.io.dataset_exists(key)):
                    value = records[key]
                    if flush:
                        # remove the record of the cached computation, and delete the filesystem cache
                        # for the computation
                        del records[key]
                        if arpes.io.dataset_exists(key):
                            arpes.io.delete_dataset(key)

                    with open(arpes.config.PIPELINE_JSON_SHELF, 'w') as file:
                        json.dump(records, file)

                    echo(value)
                    return value
                else:
                    del records[key]
                    with open(arpes.config.PIPELINE_JSON_SHELF, 'w') as file:
                        json.dump(records, file)
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
                with open(arpes.config.PIPELINE_JSON_SHELF, 'w') as file:
                    json.dump(records, file)
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
                for next_pipeline in pipelines:
                    data_in_process = next_pipeline(data_in_process, *args, **kwargs)

                return data_in_process
            except PipelineRollbackException as e:
                if not max_restarts:
                    raise e

                max_restarts -= 1
                continue

    return composed
