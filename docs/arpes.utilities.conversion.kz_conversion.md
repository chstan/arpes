# arpes.utilities.conversion.kz\_conversion module

**class
arpes.utilities.conversion.kz\_conversion.ConvertKpKzV0(\*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)

**class
arpes.utilities.conversion.kz\_conversion.ConvertKxKyKz(\*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)

**class
arpes.utilities.conversion.kz\_conversion.ConvertKpKz(\*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)
> 
> **conversion\_for(dim: str) -\> Callable**
> 
> **get\_coordinates(resolution: dict = None, bounds: dict = None) -\>
> Dict\[str, numpy.ndarray\]**
> 
> **kspace\_to\_hv(binding\_energy: numpy.ndarray, kp: numpy.ndarray,
> kz: numpy.ndarray, \*args,**kwargs) -\> numpy.ndarray\*\*
> 
> **kspace\_to\_phi(binding\_energy: numpy.ndarray, kp: numpy.ndarray,
> kz: numpy.ndarray, \*args,**kwargs) -\> numpy.ndarray\*\*
