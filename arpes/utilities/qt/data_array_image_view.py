# pylint: disable=import-error
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import weakref
from scipy import interpolate

__all__ = (
    "DataArrayImageView",
    "DataArrayPlot",
)


class CoordAxis(pg.AxisItem):
    def __init__(self, dim_index, *args, **kwargs):
        self.dim_index = dim_index
        self.coord = None
        self.interp = None
        super().__init__(*args, **kwargs)

    def setImage(self, image):
        self.coord = image.coords[image.dims[self.dim_index]].values
        self.interp = interpolate.interp1d(
            np.arange(0, len(self.coord)), self.coord, fill_value="extrapolate"
        )

    def tickStrings(self, values, scale, spacing):
        try:
            return ["{:.3f}".format(f) for f in self.interp(values)]
        except TypeError:
            return super().tickStrings(values, scale, spacing)


class DataArrayPlot(pg.PlotWidget):
    def __init__(self, root, orientation, *args, **kwargs):
        self.orientation = orientation

        axis_or = "bottom" if orientation == "horiz" else "left"
        self._coord_axis = CoordAxis(dim_index=0, orientation=axis_or)

        super().__init__(axisItems=dict([[axis_or, self._coord_axis]]), *args, **kwargs)

    def plot(self, data, *args, **kwargs):
        y = data.values
        self._coord_axis.setImage(data)

        if self.orientation == "horiz":
            self.plotItem.plot(np.arange(0, len(y)), y, *args, **kwargs)
        else:
            self.plotItem.plot(y, np.arange(0, len(y)), *args, **kwargs)


class DataArrayImageView(pg.ImageView):
    """
    ImageView that transparently handles xarray data, including setting axis and coordinate information.

    This makes it easier to build interactive applications around realistic scientific datasets.
    """

    def __init__(self, root, *args, **kwargs):
        self._coord_axes = {
            "left": CoordAxis(dim_index=1, orientation="left"),
            "bottom": CoordAxis(dim_index=0, orientation="bottom"),
        }
        plot_item = pg.PlotItem(axisItems=self._coord_axes)
        super().__init__(view=plot_item, *args, **kwargs)

        self.view.invertY(False)

    def setImage(self, img, keep_levels=False, *args, **kwargs):
        """
        Accepts an xarray.DataArray instead of a numpy array
        :param img:
        :param args:
        :param kwargs:
        :return:
        """
        if keep_levels:
            levels = self.getLevels()

        for axis in self._coord_axes.values():
            axis.setImage(img)

        super().setImage(img.values, *args, **kwargs)

        if keep_levels:
            self.setLevels(*levels)

    def recompute(self):
        pass
