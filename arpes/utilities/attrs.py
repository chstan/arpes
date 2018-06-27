import pandas as pd
import numpy as np

from arpes.typing import DataType
from pprint import pprint

__all__ = ('diff_attrs',)


def diff_attrs(a: DataType, b: DataType, should_print=True, skip_nan=False, skip_composite=True):
    attrs_a = a.attrs
    attrs_b = b.attrs

    a_has = {k: v for k, v in attrs_a.items() if k not in attrs_b}
    b_has = {k: v for k, v in attrs_b.items() if k not in attrs_a}

    def should_skip(k):
        if skip_composite:
            composites = (dict, list, np.ndarray, pd.DataFrame,)
            if isinstance(attrs_a[k], composites) or isinstance(attrs_b[k], composites):
                if type(attrs_a[k]) == type(attrs_b[k]):
                    return True

        try:
            if attrs_a[k] == attrs_b[k]:
                return True
        except ValueError:
            # probably a data frame
            return True

        if skip_nan and (np.isnan(attrs_a[k]) or np.isnan(attrs_b[k])):
            return True

        try:
            if np.isnan(attrs_a[k]) and np.isnan(attrs_b[k]):
                return True
        except:
            pass

        return False

    common = list(k for k in attrs_a.keys() if k in attrs_b and not should_skip(k))

    values_in_a = [attrs_a[k] for k in common]
    values_in_b = [attrs_b[k] for k in common]

    diff = pd.DataFrame(data={
        'key': common,
        'A': values_in_a,
        'B': values_in_b,
    }).set_index('key')


    if should_print:
        print('A has:')
        pprint(a_has)

        print('\nB has:')
        pprint(b_has)

        print('\nDifferences:')
        print(diff.to_string())
    else:
        return a_has, b_has, diff