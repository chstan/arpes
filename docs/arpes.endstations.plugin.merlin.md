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
> `ATTR_TRANSFORMS = {'acquisition_mode': ... ESEndstation.<lambda>>}`
> 
> `MERGE_ATTRS = {'analyzer': 'R8000' ... y_polarized_undulator'}`
> 
> `PRINCIPAL_NAME = 'ALS-BL403'`
> 
> `RENAME_KEYS = {'BL Energy': 'hv', ... 'user':
> 'experimenter'}`
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
