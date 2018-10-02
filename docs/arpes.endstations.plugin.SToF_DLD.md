# arpes.endstations.plugin.SToF\_DLD module

**class arpes.endstations.plugin.SToF\_DLD.SToFDLDEndstation**

> Bases:
> [arpes.endstations.EndstationBase](arpes.endstations#arpes.endstations.EndstationBase)
> 
> `PRINCIPAL_NAME = 'ALG-SToF-DLD'`
> 
> **load(scan\_desc: dict = None,**kwargs)\*\*
> 
> > Imports a FITS file that contains all of the information from a run
> > of Ping and Anton’s delay line detector ARToF
> > 
> >   - Parameters  
> >     **scan\_desc** – Dictionary with extra information to attach to
> >     the xarray.Dataset, must contain the location
> > 
> > of the file :return: xarray.Dataset
