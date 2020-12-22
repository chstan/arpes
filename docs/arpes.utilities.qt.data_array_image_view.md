arpes.utilities.qt.data\_array\_image\_view module
==================================================

**class
arpes.utilities.qt.data\_array\_image\_view.DataArrayImageView(root,
\*args,**kwargs)\*\*

> Bases: `pyqtgraph.imageview.ImageView.ImageView`
>
> ImageView that transparently handles xarray data, including setting
> axis and coordinate information.
>
> This makes it easier to build interactive applications around
> realistic scientific datasets.
>
> **recompute()**
>
> **setImage(img, keep\_levels=False, \*args,**kwargs)\*\*
>
> > Accepts an xarray.DataArray instead of a numpy array :param img:
> > :param args: :param kwargs: :return:

**class arpes.utilities.qt.data\_array\_image\_view.DataArrayPlot(root,
orientation, \*args,**kwargs)\*\*

> Bases: `pyqtgraph.widgets.PlotWidget.PlotWidget`
>
> **plot(data, \*args,**kwargs)\*\*
