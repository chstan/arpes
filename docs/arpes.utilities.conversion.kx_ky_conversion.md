# arpes.utilities.conversion.kx\_ky\_conversion module

**class
arpes.utilities.conversion.kx\_ky\_conversion.ConvertKp(\*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)
> 
> **compute\_k\_tot(binding\_energy: numpy.ndarray) -\> None**
> 
> **conversion\_for(dim: str) -\> Callable**
> 
> **get\_coordinates(resolution: dict = None, bounds: dict = None) -\>
> Dict\[str, numpy.ndarray\]**
> 
> **kspace\_to\_phi(binding\_energy: numpy.ndarray, kp: numpy.ndarray,
> \*args,**kwargs) -\> numpy.ndarray\*\*

**class arpes.utilities.conversion.kx\_ky\_conversion.ConvertKxKy(arr:
xarray.core.dataarray.DataArray, \*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)
> 
> Please note that currently we assume that psi = 0 when you are not
> using an electrostatic deflector
> 
> **compute\_k\_tot(binding\_energy: numpy.ndarray) -\> None**
> 
> **conversion\_for(dim: str) -\> Callable**
> 
> **get\_coordinates(resolution: dict = None, bounds: dict = None) -\>
> Dict\[str, numpy.ndarray\]**
> 
> **kspace\_to\_perp\_angle(binding\_energy: numpy.ndarray, kx:
> numpy.ndarray, ky: numpy.ndarray, \*args,**kwargs) -\>
> numpy.ndarray\*\*
> 
> **kspace\_to\_phi(binding\_energy: numpy.ndarray, kx: numpy.ndarray,
> ky: numpy.ndarray, \*args,**kwargs) -\> numpy.ndarray\*\*
> 
> `needs_rotation`
> 
> **rkx\_rky(kx, ky)**
> 
> > Returns the rotated kx and ky values when we are rotating by nonzero
> > chi :return:
