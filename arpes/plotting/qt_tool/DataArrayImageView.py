import pyqtgraph as pg

__all__ = ('DataArrayImageView',)

class DataArrayImageView(pg.ImageView):
    """
    ImageView that transparently handles xarray data, including setting axis and coordinate information.

    This makes it easier to build interactive applications around realistic scientific datasets.
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__(view=pg.PlotItem(), *args, **kwargs)

        self.view.invertY(False)
        self.root = root

    def setImage(self, img, *args, **kwargs):
        """
        Accepts an xarray.DataArray instead of a numpy array
        :param img:
        :param args:
        :param kwargs:
        :return:
        """

        super().setImage(img.values, *args, **kwargs)

    def recompute(self):
        pass

