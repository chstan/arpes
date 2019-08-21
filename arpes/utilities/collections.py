import collections

import numpy as np

from typing import Any, Dict

__all__ = ('deep_equals', 'deep_update', 'MappableDict',)


class MappableDict(dict):
    """
    Like dict except that +, -, *, / are cascaded to values.
    """

    def __add__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise ValueError('You can only add two MappableDicts with the same keys.')

        return MappableDict({
            k: self.get(k) + other.get(k) for k in self.keys()
        })

    def __sub__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise ValueError('You can only subtract two MappableDicts with the same keys.')

        return MappableDict({
            k: self.get(k) - other.get(k) for k in self.keys()
        })

    def __mul__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise ValueError('You can only multiply two MappableDicts with the same keys.')

        return MappableDict({
            k: self.get(k) * other.get(k) for k in self.keys()
        })

    def __truediv__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise ValueError('You can only divide two MappableDicts with the same keys.')

        return MappableDict({
            k: self.get(k) / other.get(k) for k in self.keys()
        })

    def __floordiv__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise ValueError('You can only divide (//) two MappableDicts with the same keys.')

        return MappableDict({
            k: self.get(k) // other.get(k) for k in self.keys()
        })

    def __neg__(self):
        return MappableDict({
            k: -self.get(k) for k in self.keys()
        })


def deep_update(destination: Any, source: Any) -> Dict[str, Any]:
    """
    Doesn't clobber keys further down trees like doing a shallow update would. Instead recurse down from the root
    and update as appropriate.
    :param destination:
    :param source:
    :return:
    """
    for k, v in source.items():
        if isinstance(v, collections.Mapping):
            destination[k] = deep_update(destination.get(k, {}), v)
        else:
            destination[k] = v

    return destination


def deep_equals(a: Any, b: Any) -> bool:
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
