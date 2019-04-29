"""
Provides a jumping off point for defining data loading plugins using the NeXuS file format.
Currently we assume that the raw file format is actually HDF.
"""

__all__ = ('read_data_attributes_from',)


def read_data_attributes_from(group, paths):
    read_attrs = {}
    original_group = group
    for path, attributes in paths:
        group = original_group
        for p in path:
            group = group[p]

        for attribute_name in attributes:
            try:
                data = group[attribute_name]['data']
            except ValueError:
                data = group[attribute_name]
            try:
                data = data[:]
            except ValueError:
                data = data.value

            read_attrs[attribute_name] = data

    return read_attrs
