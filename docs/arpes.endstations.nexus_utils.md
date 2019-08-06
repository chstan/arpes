# arpes.endstations.nexus\_utils module

Provides a jumping off point for defining data loading plugins using the
NeXuS file format. Currently we assume that the raw file format is
actually HDF.

**arpes.endstations.nexus\_utils.read\_data\_attributes\_from(group,
paths)**

> Reads simple (float, string, int, etc) leaves from a nested
> description of paths out of a NeXuS file. :param group: :param paths:
> :return:
