"""Defines a widget which provides a 1D browsable `lmfit.model.ModelResult`."""

from PyQt5 import QtCore, QtWidgets
from arpes.utilities.qt.data_array_image_view import DataArrayPlot
from typing import Optional
import pyqtgraph as pg
import xarray as xr
import numpy as np
import math
from PyQt5.QtWidgets import QLayout, QGridLayout, QWidget
import lmfit
from weakref import ReferenceType

from arpes.utilities.qt import qt_info

__all__ = ["FitInspectionPlot"]


class LabelParametersInfoView(QtWidgets.QLabel):
    def __init__(self, parent=None):
        """Nothing interesting to do here, we just delegate to super."""
        super().__init__(parent)
        self.setText("")

    def set_model_result(self, model_result: lmfit.model.ModelResult):
        """Converts the ModelResult to the HTML representation and sets page contents."""
        assert model_result is not None
        self.setText(model_result._repr_multiline_text_(short=True))


class FitInspectionPlot(QWidget):
    """Implements the `ModeResult` inspection tool."""

    layout: QLayout = None
    result: Optional[lmfit.model.ModelResult] = None
    root: ReferenceType = None

    def x(self) -> np.ndarray:
        """Returns the single fit coordinate along the model data."""
        fit_dim = self.root().dataset.F.fit_dimensions[0]
        return self.root().dataset.coords[fit_dim].values

    @property
    def data(self) -> np.ndarray:
        """Returns the values of the data used for fitting."""
        return self.result.data

    @property
    def residual(self) -> np.ndarray:
        """Returns the residual from the fit."""
        return self.result.residual

    @property
    def eval_model(self) -> np.ndarray:
        """Returns the values of the fit at the x coordinates.

        Rather than accomplishing this by actually calling the model
        eval, we cheat by using the original data and the residual to back out
        the fit data.
        """
        return self.data - self.residual

    @property
    def init_eval_model(self) -> np.ndarray:
        """Returns the initial values of the fit.

        This is useful to see coarsely whether the initial parameter
        values being used are leading to reasonable and convergent fits.
        """
        return self.result.init_fit

    def set_model_result(self, model_result: lmfit.model.ModelResult) -> None:
        """Updates the current widgets according to the new coords/model result.

        In order to do this, we need to update the three lineplots on the plot widget:
        * The data
        * The residual
        * The evaled data
        * (TODO, the component data)

        and we need to update the model info section.
        """
        # assign the model result onto this instance
        self.result = model_result

        # update the plotted data
        x = self.x()
        coords = {"x": x}
        dims = ["x"]

        p1 = self.inner_plot.plot(xr.DataArray(self.data, coords, dims), clear=True)
        p2 = self.inner_plot.plot(xr.DataArray(self.residual, coords, dims))
        p3 = self.inner_plot.plot(xr.DataArray(self.eval_model, coords, dims))
        p4 = self.inner_plot.plot(xr.DataArray(self.init_eval_model, coords, dims))

        plot_width = int(math.ceil(qt_info.inches_to_px(0.02)))
        p1.setPen(pg.mkPen(width=plot_width, color=(0, 0, 0)))
        p2.setPen(pg.mkPen(width=plot_width, color=(255, 0, 0)))
        p3.setPen(pg.mkPen(width=plot_width, color=(50, 200, 20), style=QtCore.Qt.DotLine))
        p4.setPen(pg.mkPen(width=plot_width, color=(70, 150, 70), style=QtCore.Qt.DotLine))

        # update the model info
        self.model_info.set_model_result(model_result)

    def close(self):
        """Clean up references so we do not cause GC issues and Qt crashes."""
        self.model_info.close()
        self.inner_plot.close()
        self.model_info = None
        self.inner_plot = None
        self.root = None
        self.result = None

        super().close()

    def __init__(self, root, orientation, name=None, *args, **kwargs):
        """Performs initial registration of the widgets and sets up layout."""
        super().__init__(*args, **kwargs)
        self.layout = QGridLayout()
        self.model_info = LabelParametersInfoView()
        self.inner_plot = DataArrayPlot(root, orientation, name=name)

        self.setLayout(self.layout)
        self.layout.addWidget(self.inner_plot, 0, 0)
        self.layout.addWidget(self.model_info, 0, 1)

        self.root = root
