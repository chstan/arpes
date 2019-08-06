# arpes.analysis.xps module

**arpes.analysis.xps.approximate\_core\_levels(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
window\_size=None, order=5, binning=3, promenance=5)**

> Approximately locates core levels in a spectrum. Data is first
> smoothed, and then local maxima with sufficient prominence over other
> nearby points are selected as peaks.
> 
> This can be helfpul to “seed” a curve fitting analysis for XPS. :param
> data: :param window\_size: :param order: :param binning: :param
> promenance: :return:
