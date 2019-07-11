# arpes.endstations.plugin.MAESTRO module

**class arpes.endstations.plugin.MAESTRO.MAESTROARPESEndstation**

> Bases:
> [arpes.endstations.SynchrotronEndstation](arpes.endstations#arpes.endstations.SynchrotronEndstation),
> [arpes.endstations.HemisphericalEndstation](arpes.endstations#arpes.endstations.HemisphericalEndstation),
> [arpes.endstations.FITSEndstation](arpes.endstations#arpes.endstations.FITSEndstation)
> 
> The MERLIN ARPES Endstation at the Advanced Light Source
> 
> `ALIASES = ['BL7', 'BL7.0.2', 'ALS-BL7.0.2']`
> 
> `PRINCIPAL_NAME = 'ALS-BL702'`
> 
> `RENAME_KEYS = {'LMOTOR0': 'x', 'LM ... olar', 'mono_eV': 'hv'}`
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
