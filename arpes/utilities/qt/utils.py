"""Contains utility classes for Qt in PyARPES."""

import enum
from dataclasses import dataclass
from typing import List
from PyQt5 import QtWidgets

__all__ = ["PlotOrientation", "ReactivePlotRecord"]


class PlotOrientation(str, enum.Enum):
    """Controls the transposition on a reactive plot."""

    Horizontal = "horizontal"
    Vertical = "vertical"


@dataclass
class ReactivePlotRecord:
    """This contains metadata related to a reactive plot or marginal on a DataArary.

    This is used to know how to update and mount corresponding widgets on a main tool view.
    """

    dims: List[str]
    view: QtWidgets.QWidget
    orientation: PlotOrientation
