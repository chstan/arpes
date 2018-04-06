import numpy as np
from collections import namedtuple

ParamType = namedtuple('ParamType', ['value', 'stderr'])


def param_getter(param_name, safe=True):
    if safe:
        safe_param = ParamType(value=np.nan, stderr=np.nan)

        def getter(x):
            try:
                return x.params.get(param_name, safe_param).value
            except:
                return np.nan

        return getter

    return lambda x: x.params[param_name].value


def param_stderr_getter(param_name, safe=True):
    if safe:
        safe_param = ParamType(value=np.nan, stderr=np.nan)

        def getter(x):
            try:
                return x.params.get(param_name, safe_param).stderr
            except:
                return np.nan

        return getter

    return lambda x: x.params[param_name].stderr