# arpes.endstations.plugin.ANTARES module

**class arpes.endstations.plugin.ANTARES.ANTARESEndstation**

> Bases:
> [arpes.endstations.HemisphericalEndstation](arpes.endstations#arpes.endstations.HemisphericalEndstation),
> [arpes.endstations.SynchrotronEndstation](arpes.endstations#arpes.endstations.SynchrotronEndstation),
> [arpes.endstations.SingleFileEndstation](arpes.endstations#arpes.endstations.SingleFileEndstation)
> 
> Implements loading text files from the MB Scientific text file format.
> 
> There’s not too much metadata here except what comes with the analyzer
> settings.
> 
> `ALIASES = []`
> 
> `PRINCIPAL_NAME = 'ANTARES'`
> 
> `RENAME_KEYS = {'deflx': 'polar'}`
> 
> **get\_coords(group, scan\_name, shape)**
> 
> **load\_single\_frame(frame\_path: str = None, scan\_desc: dict =
> None,**kwargs)\*\*
> 
> **load\_top\_level\_scan(group, scan\_desc: dict = None,
> spectrum\_index=None)**
> 
> **read\_scan\_data(group)**
> 
> > Reads the scan data stored in /scan\_data/[data](){idx} as
> > appropriate for the type of file.
