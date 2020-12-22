arpes.endstations.plugin.fallback module
========================================

**class arpes.endstations.plugin.fallback.FallbackEndstation**

> Bases:
> [arpes.endstations.EndstationBase](arpes.endstations#arpes.endstations.EndstationBase)
>
> Different from the rest of the data loaders. This one is used when
> there is no location specified and attempts sequentially to call a
> variety of standard plugins until one is found that works.
>
> `ALIASES = []`
>
> `ATTEMPT_ORDER = ['ANTARES', 'MBS', ' ... 'ALG-Main', 'ALG-SToF']`
>
> `PRINCIPAL_NAME = 'fallback'`
>
> **classmethod determine\_associated\_loader(file, scan\_desc)**
>
> **classmethod find\_first\_file(file, scan\_desc,
> allow\_soft\_match=False)**
>
> **load(scan\_desc: Optional\[dict\] = None, file=None,**kwargs)\*\*
>
> > Loads a scan from a single file or a sequence of files.
> >
> > Parameters  
> > -   **scan\_desc** –
> > -   **kwargs** –
> >
> > Returns  
