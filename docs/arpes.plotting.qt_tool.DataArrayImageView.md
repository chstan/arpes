# arpes.plotting.qt\_tool.DataArrayImageView module

**class
arpes.plotting.qt\_tool.DataArrayImageView.DataArrayImageView(root,
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
> **setImage(img, \*args,**kwargs)\*\*
> 
> > Accepts an xarray.DataArray instead of a numpy array :param img:
> > :param args: :param kwargs: :return:
