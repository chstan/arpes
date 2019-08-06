# arpes.endstations.plugin.MBS module

**class arpes.endstations.plugin.MBS.MBSEndstation**

> Bases:
> [arpes.endstations.HemisphericalEndstation](arpes.endstations#arpes.endstations.HemisphericalEndstation)
> 
> Implements loading text files from the MB Scientific text file format.
> 
> There’s not too much metadata here except what comes with the analyzer
> settings.
> 
> `ALIASES = ['MB Scientific']`
> 
> `PRINCIPAL_NAME = 'MBS'`
> 
> `RENAME_KEYS = {'deflx': 'psi'}`
> 
> **load\_single\_frame(frame\_path: str = None, scan\_desc: dict =
> None,**kwargs)\*\*
> 
> **postprocess\_final(data: xarray.core.dataset.Dataset, scan\_desc:
> dict = None)**
> 
> **resolve\_frame\_locations(scan\_desc: dict = None)**
