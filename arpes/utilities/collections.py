import numpy as np

__all__ = ('deep_equals',)


def deep_equals(a, b):
    if not isinstance(b, type(a)):
        print(b, a)
        return False

    if isinstance(a, (int, str, float, np.float, np.int, np.float64, np.int64,),):
        return a == b

    if a is None:
        return b is None

    if not isinstance(a, (dict, list, tuple, set,)):
        raise TypeError('Only dict, list, tuple, and set are supported by deep_equals, not {}'.format(type(a)))

    if isinstance(a, set):
        for item in a:
            if item not in b:
                return False

        for item in b:
            if item not in a:
                return False

        return True

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False

        for i in range(len(a)):
            item_a, item_b = a[i], b[i]

            if not deep_equals(item_a, item_b):
                return False

        return True

    if isinstance(a, dict):
        if not set(a.keys()) == set(b.keys()):
            return False

        for k in a.keys():
            item_a, item_b = a[k], b[k]

            if not deep_equals(item_a, item_b):
                return False

        return True