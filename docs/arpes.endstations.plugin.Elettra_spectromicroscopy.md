arpes.endstations.plugin.Elettra\_spectromicroscopy module
==========================================================

**class
arpes.endstations.plugin.Elettra\_spectromicroscopy.SpectromicroscopyElettraEndstation**

> Bases:
> [arpes.endstations.HemisphericalEndstation](arpes.endstations#arpes.endstations.HemisphericalEndstation),
> [arpes.endstations.SynchrotronEndstation](arpes.endstations#arpes.endstations.SynchrotronEndstation)
>
> Data loading for the nano-ARPES beamline “Spectromicroscopy Elettra”.
>
> Information available on the beamline can be accessed
> [here](https://www.elettra.trieste.it/elettra-beamlines/spectromicroscopy).
>
> `ALIASES = ['Spectromicroscopy', 'nano-ARPES Elettra']`
>
> `ANALYZER_INFORMATION = {'analyzer': 'Custom ... lar_deflectors': False}`
>
> `CONCAT_COORDS = ['T', 'P']`
>
> `PRINCIPAL_NAME = 'Spectromicroscopy Elettra'`
>
> `RENAME_COORDS = {'Angle': 'phi', 'KE ... x', 'Y': 'y', 'Z': 'z'}`
>
> `RENAME_KEYS = {'Dwell Time (s)': ' ... re (K)': 'temperature'}`
>
> **concatenate\_frames(frames=typing.List\[xarray.core.dataset.Dataset\],
> scan\_desc: Optional\[dict\] = None)**
>
> **classmethod files\_for\_search(directory)**
>
> **load\_single\_frame(frame\_path: Optional\[str\] = None, scan\_desc:
> Optional\[dict\] = None,**kwargs)\*\*
>
> **postprocess\_final(data: xarray.core.dataset.Dataset, scan\_desc:
> Optional\[dict\] = None)**
>
> **resolve\_frame\_locations(scan\_desc: Optional\[dict\] = None)**
