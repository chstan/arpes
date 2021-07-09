"""Utilities related to fixing quirks in data."""

from arpes.typing import DataType

__all__ = ["negate_energy"]


def negate_energy(data: DataType) -> DataType:
    """Negates the energy coordinate on a piece of data.

    This is useful in changing between KE/BE conventions when data is loaded.

    Args:
        data: The piece of data which we should negate the energy axis of.

    Returns:
        The data with the updated energy coordinate.
    """
    if "eV" in data.coords:
        data = data.assign_coords(eV=-data.eV.values)

    return data
