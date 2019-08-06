import re

from arpes.utilities.xarray import lift_dataarray_attrs, lift_datavar_attrs

__all__ = ('rename_keys', 'clean_keys', 'rename_dataarray_attrs', 'rename_datavar_attrs',
           'clean_datavar_attribute_names', 'clean_attribute_names', 'case_insensitive_get')

def _rename_key(d, k, nk):
    if k in d:
        d[nk] = d[k]
        del d[k]


def rename_keys(d, keys_dict):
    d = d.copy()
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d


def clean_keys(d):
    def clean_single_key(k):
        k = k.replace(' ', '_')
        k = k.replace('.', '_')
        k = k.lower()
        k = re.sub(r'[()/?]', '', k)
        k = k.replace('__', '_')
        return k

    return dict(zip([clean_single_key(k) for k in d.keys()], d.values()))


def case_insensitive_get(d: dict, key: str, default=None, take_first=False):
    """
    Looks up a key in a dictionary ignoring case. We use this sometimes to be
    nicer to users who don't provide perfectly sanitized data
    :param d:
    :param key:
    :param default:
    :param take_first:
    :return:
    """
    found_value = False
    value = None

    for k, v in d.items():
        if k.lower() == key.lower():
            if not take_first and found_value:
                raise ValueError('Duplicate case insensitive keys')

            value = v
            found_value = True

            if take_first:
                break

    if not found_value:
        return default

    return value


rename_dataarray_attrs = lift_dataarray_attrs(rename_keys)
clean_attribute_names = lift_dataarray_attrs(clean_keys)

rename_datavar_attrs = lift_datavar_attrs(rename_keys)
clean_datavar_attribute_names = lift_datavar_attrs(clean_keys)
