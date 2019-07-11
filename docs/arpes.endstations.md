# arpes.endstations package

## Subpackages

  -   - [arpes.endstations.plugin package](arpes.endstations.plugin)
        
          -   - [Submodules](arpes.endstations.plugin#submodules)
                
                  - [arpes.endstations.plugin.ALG\_main
                    module](arpes.endstations.plugin.ALG_main)
                  - [arpes.endstations.plugin.ALG\_spin\_ToF
                    module](arpes.endstations.plugin.ALG_spin_ToF)
                  - [arpes.endstations.plugin.ANTARES
                    module](arpes.endstations.plugin.ANTARES)
                  - [arpes.endstations.plugin.HERS
                    module](arpes.endstations.plugin.HERS)
                  - [arpes.endstations.plugin.MAESTRO
                    module](arpes.endstations.plugin.MAESTRO)
                  - [arpes.endstations.plugin.MBS
                    module](arpes.endstations.plugin.MBS)
                  - [arpes.endstations.plugin.SToF\_DLD
                    module](arpes.endstations.plugin.SToF_DLD)
                  - [arpes.endstations.plugin.igor\_export
                    module](arpes.endstations.plugin.igor_export)
                  - [arpes.endstations.plugin.kaindl
                    module](arpes.endstations.plugin.kaindl)
                  - [arpes.endstations.plugin.merlin
                    module](arpes.endstations.plugin.merlin)
        
          - [Module
            contents](arpes.endstations.plugin#module-arpes.endstations.plugin)

## Submodules

  - [arpes.endstations.fits\_utils module](arpes.endstations.fits_utils)
  - [arpes.endstations.igor\_utils module](arpes.endstations.igor_utils)
  - [arpes.endstations.nexus\_utils
    module](arpes.endstations.nexus_utils)

## Module contents

Plugin facility to read+normalize information from different sources to
a common format

**arpes.endstations.endstation\_name\_from\_alias(alias)**

**arpes.endstations.endstation\_from\_alias(alias)**

**arpes.endstations.add\_endstation(endstation\_cls)**

**arpes.endstations.load\_scan(scan\_desc,**kwargs)\*\*

**class arpes.endstations.EndstationBase**

> Bases: `object`
> 
> `ALIASES = []`
> 
> `CONCAT_COORDS = ['hv', 'polar', 'timed_power', 'tilt']`
> 
> `PRINCIPAL_NAME = None`
> 
> `RENAME_KEYS = {}`
> 
> `SUMMABLE_NULL_DIMS = ['phi',
> 'cycle']`
> 
> **concatenate\_frames(frames=typing.List\[xarray.core.dataset.Dataset\],
> scan\_desc: dict = None)**
> 
> **load(scan\_desc: dict = None,**kwargs)\*\*
> 
> > Loads a scan from a single file or a sequence of files.
> > 
> >   - Parameters
> >     
> >       - **scan\_desc** –
> >       - **kwargs** –
> > 
> >   - Returns
> 
> **load\_single\_frame(frame\_path: str = None, scan\_desc: dict =
> None,**kwargs)\*\*
> 
> **postprocess(frame: xarray.core.dataset.Dataset)**
> 
> **postprocess\_final(data: xarray.core.dataset.Dataset, scan\_desc:
> dict = None)**
> 
> **resolve\_frame\_locations(scan\_desc: dict = None)**

**class arpes.endstations.FITSEndstation**

> Bases:
> 
> `PREPPED_COLUMN_NAMES = {'Delay': 'delay-var ... e-var', 'time':
> 'time'}`
> 
> `RENAME_KEYS = {'Azimuth': 'chi', ' ... _func': 'workfunction'}`
> 
> `SKIP_COLUMN_FORMULAS = {<function FITSEndstation.<lambda>>}`
> 
> `SKIP_COLUMN_NAMES = {'Optics Stage', 'Ph ... 'Z', 'mono_eV', 'null'}`
> 
> **load\_single\_frame(frame\_path: str = None, scan\_desc: dict =
> None,**kwargs)\*\*
> 
> **resolve\_frame\_locations(scan\_desc: dict = None)**

**class arpes.endstations.HemisphericalEndstation**

> Bases:
> 
> An endstation definition for a hemispherical analyzer should include
> everything needed to determine energy + k resolution, angle
> conversion, and ideally correction databases for dead pixels +
> detector nonlinearity information
> 
> `ANALYZER_INFORMATION = None`
> 
> `PIXELS_PER_DEG = None`
> 
> `SLIT_ORIENTATION = None`

**class arpes.endstations.SynchrotronEndstation**

> Bases:
> 
> `RESOLUTION_TABLE = None`

**class arpes.endstations.SingleFileEndstation**

> Bases:
> 
> **resolve\_frame\_locations(scan\_desc: dict = None)**

**arpes.endstations.load\_scan\_for\_endstation(scan\_desc,
endstation\_cls,**kwargs)\*\*
