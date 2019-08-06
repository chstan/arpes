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

> Lookup the data loading principal location from an alias. :param
> alias: :return:

**arpes.endstations.endstation\_from\_alias(alias)**

> Lookup the data loading class from an alias. :param alias: :return:

**arpes.endstations.add\_endstation(endstation\_cls)**

> Registers a data loading plugin (Endstation class) together with its
> aliases.
> 
>   - Parameters  
>     **endstation\_cls** –
> 
>   - Returns

**arpes.endstations.load\_scan(scan\_desc,**kwargs)\*\*

> Determines which data loading class is appropriate for the data,
> shuffles a bit of metadata, and calls the .load function on the
> retrieved class to start the data loading process. :param scan\_desc:
> :param kwargs: :return:

**class arpes.endstations.EndstationBase**

> Bases: `object`
> 
> Implements the core features of ARPES data loading. A thorough
> documentation is available at [the plugin
> documentation](https://arpes.netlify.com/#/writing-plugins).
> 
> To summarize, a plugin has a few core jobs:
> 
> 1.    - Load data, including collating any data that is in a
>         multi-file  
>         format This is accomplished with *.load*, which delegates
>         loading *frames* (single files) to *load\_single\_frame*.
>         Frame collation is then performed by *concatenate\_frames*.
> 
> 2.  Loading and attaching metadata.
> 
> 3.    - Normalizing metadata to standardized names. These are  
>         documented at the [data model
>         documentation](https://arpes.netlify.com/#/spectra).
> 
> 4.    - Ensuring all angles and necessary coordinates are attached
>         to  
>         the data. Data should permit immediate conversion to angle
>         space after being loaded.
> 
> Plugins are in one-to-many correspondance with the values of the
> “location” column in analysis spreadsheets. This binding is provided
> by PRINCIPAL\_NAME and ALIASES.
> 
> The simplest way to normalize metadata is by renaming keys, but
> sometimes additional work is required. RENAME\_KEYS is provided to
> make this simpler, and is implemented in scan post-processessing.
> 
> `ALIASES = []`
> 
> `ATTR_TRANSFORMS = {}`
> 
> `CONCAT_COORDS = ['hv', 'chi', 'psi', 'timed_power', 'tilt', 'beta',
> 'theta']`
> 
> `MERGE_ATTRS = {}`
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
> Loads data from the .fits format produced by the MAESTRO software and
> derivatives.
> 
> This ends up being somewhat complicated, because the FITS export is
> written in LabView and does not conform to the standard specification
> for the FITS archive format.
> 
> Many of the intricaces here are in fact those shared between MAESTRO’s
> format and the Lanzara Lab’s format. Conrad does not forsee this as an
> issue, because it is unlikely that mnay other ARPES labs will adopt
> this data format moving forward, in light of better options derivative
> of HDF like the NeXuS format.
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
> Synchrotron endstations have somewhat complicated light source
> metadata. This stub exists to attach commonalities, such as a
> resolution table which can be interpolated into to retrieve the x-ray
> linewidth at the experimental settings. Additionally, subclassing this
> is used in resolution calculations to signal that such a resolution
> lookup is required.
> 
> `RESOLUTION_TABLE = None`

**class arpes.endstations.SingleFileEndstation**

> Bases:
> 
> Abstract endstation which loads data from a single file. This just
> specializes the routine used to determine the location of files on
> disk.
> 
> Unlike general endstations, if your data comes in a single file you
> can trust that the file given to you in the spreadsheet or direct load
> calls is all there is.
> 
> **resolve\_frame\_locations(scan\_desc: dict = None)**

**arpes.endstations.load\_scan\_for\_endstation(scan\_desc,
endstation\_cls,**kwargs)\*\*
