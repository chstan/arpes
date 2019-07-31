# arpes.utilities.conversion.kx\_ky\_conversion module

**class
arpes.utilities.conversion.kx\_ky\_conversion.ConvertKp(\*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)
> 
> **compute\_k\_tot(binding\_energy)**
> 
> **conversion\_for(dim)**
> 
> **get\_coordinates(resolution: dict = None)**
> 
> **kspace\_to\_phi(binding\_energy, kp, \*args,**kwargs)\*\*

**class arpes.utilities.conversion.kx\_ky\_conversion.ConvertKxKy(arr,
\*args,**kwargs)\*\*

> Bases:
> [arpes.utilities.conversion.base.CoordinateConverter](arpes.utilities.conversion.base#arpes.utilities.conversion.base.CoordinateConverter)
> 
> Please note that currently we assume that psi = 0 when you are not
> using an electrostatic deflector
> 
> **compute\_k\_tot(binding\_energy)**
> 
> **conversion\_for(dim)**
> 
> **get\_coordinates(resolution: dict = None)**
> 
> **kspace\_to\_perp\_angle(binding\_energy, kx, ky,
> \*args,**kwargs)\*\*
> 
> **kspace\_to\_phi(binding\_energy, kx, ky, \*args,**kwargs)\*\*
> 
> `needs_rotation`
> 
> **rkx\_rky(kx, ky)**
> 
> > Returns the rotated kx and ky values when we are rotating by nonzero
> > chi :return:
