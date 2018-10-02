# arpes.endstations.plugin.HERS module

**class arpes.endstations.plugin.HERS.HERSEndstation**

> Bases:
> [arpes.endstations.SynchrotronEndstation](arpes.endstations#arpes.endstations.SynchrotronEndstation),
> [arpes.endstations.HemisphericalEndstation](arpes.endstations#arpes.endstations.HemisphericalEndstation)
> 
> This should be unified with the FITs endstation code, but I don’t have
> any projects at BL10 at the moment so I will defer the complexity of
> unifying them for now
> 
> `ALIASES = ['ALS-BL1001', 'HERS', 'ALS-HERS', 'BL1001']`
> 
> `PRINCIPAL_NAME = 'ALS-BL1001'`
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
