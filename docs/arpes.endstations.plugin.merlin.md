# arpes.endstations.plugin.merlin module

**class arpes.endstations.plugin.merlin.BL403ARPESEndstation**

> Bases:
> [arpes.endstations.SynchrotronEndstation](arpes.endstations#arpes.endstations.SynchrotronEndstation),
> [arpes.endstations.HemisphericalEndstation](arpes.endstations#arpes.endstations.HemisphericalEndstation),
> `arpes.endstations.SESEndstation`
> 
> The MERLIN ARPES Endstation at the Advanced Light Source
> 
> `ALIASES = ['BL403', 'BL4', 'BL4.0.3', 'ALS-BL403', 'ALS-BL4']`
> 
> `PRINCIPAL_NAME = 'ALS-BL403'`
> 
> `RENAME_KEYS = {'Azimuth': 'chi', ' ... ture_sensor_b':
> 'temp'}`
> 
> **concatenate\_frames(frames=typing.List\[xarray.core.dataset.Dataset\],
> scan\_desc: dict = None)**
> 
> **load\_single\_frame(frame\_path: str = None, scan\_desc: dict =
> None,**kwargs)\*\*
> 
> **load\_single\_region(region\_path: str = None, scan\_desc: dict =
> None,**kwargs)\*\*
> 
> **postprocess\_final(data: xarray.core.dataset.Dataset, scan\_desc:
> dict = None)**
