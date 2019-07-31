# arpes.utilities.conversion.base module

**class arpes.utilities.conversion.base.CoordinateConverter(arr:
xarray.core.dataarray.DataArray, dim\_order=None, \*args,**kwargs)\*\*

> Bases: `object`
> 
> **conversion\_for(dim)**
> 
> **get\_coordinates(resolution: dict = None)**
> 
> **identity\_transform(axis\_name, \*args,**kwargs)\*\*
> 
> `is_slit_vertical`
> 
> **kspace\_to\_BE(binding\_energy, \*args,**kwargs)\*\*
> 
> **prep(arr: xarray.core.dataarray.DataArray)**
> 
> > The CoordinateConverter.prep method allows you to pre-compute some
> > transformations that are common to the individual coordinate
> > transform methods as an optimization.
> > 
> > This is useful if you want the conversion methods to have separation
> > of concern, but if it is advantageous for them to be able to share a
> > computation of some variable. An example of this is in BE-kx-ky
> > conversion, where computing k\_p\_tot is a step in both converting
> > kx and ky, so we would like to avoid doing it twice.
> > 
> > Of course, you can neglect this function entirely. Another technique
> > is to simple cache computations as they arrive. This is the
> > technique that is used in ConvertKxKy below :param arr: :return:
