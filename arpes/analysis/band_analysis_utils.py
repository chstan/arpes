from collections import namedtuple

import numpy as np

ParamType = namedtuple('ParamType', ['value', 'stderr'])


def param_getter(param_name, safe=True):
    """
    Constructs a function to extract a parameter value by name. Useful to extract data from inside an array of
    `lmfit.ModelResult` instances.

    :param param_name: Parameter name to retrieve. If you performed a composite model fit,
    make sure to include the prefix.
    :param safe: Guards against NaN values. This is typically desirable but sometimes it is advantageous
    to have NaNs fail an analysis quickly.
    :return:
    """
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
    """
    Constructs a function to extract a parameter value by name. Useful to extract data from inside an array of
    `lmfit.ModelResult` instances.

    :param param_name: Parameter name to retrieve. If you performed a composite model fit,
    make sure to include the prefix.
    :param safe: Guards against NaN values. This is typically desirable but sometimes it is advantageous
    to have NaNs fail an analysis quickly.
    :return:
    """

    if safe:
        safe_param = ParamType(value=np.nan, stderr=np.nan)

        def getter(x):
            try:
                return x.params.get(param_name, safe_param).stderr
            except:
                return np.nan

        return getter

    return lambda x: x.params[param_name].stderr
