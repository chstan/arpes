# arpes.typing module

Specialized type annotations for use in PyARPES.

In particular, we frequently allow using the *DataType* annotation,
which refers to either an xarray.DataArray or xarray.Dataset.

Additionally, we often use *NormalizableDataType* which means
essentially anything that can be turned into a dataset, for instance by
loading from the cache using an ID, or which is literally already data.
