import uuid

import xarray as xr

import typing

__all__ = ['DataType', 'NormalizableDataType', 'xr_types']

DataType = typing.Union[xr.DataArray, xr.Dataset]
NormalizableDataType = typing.Union[DataType, str, uuid.UUID]

xr_types = (xr.DataArray, xr.Dataset,)