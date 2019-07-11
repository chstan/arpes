# arpes.endstations.plugin.ALG\_spin\_ToF module

**class arpes.endstations.plugin.ALG\_spin\_ToF.SpinToFEndstation**

> Bases:
> [arpes.endstations.EndstationBase](arpes.endstations#arpes.endstations.EndstationBase)
> 
> `ALIASES = ['ALG-SToF', 'SToF', 'Spin-ToF', 'ALG-SpinToF']`
> 
> `COLUMN_RENAMINGS = {'ALS_Beam_mA': 'bea ... p', 'wave': 'spectrum'}`
> 
> `PRINCIPAL_NAME = 'ALG-SToF'`
> 
> `RENAME_KEYS = {'LMOTOR0': 'x', 'LM ... 'delay', 'Phi': 'phi'}`
> 
> `SKIP_ATTR_FRAGMENTS = {'BITPIX', 'COMMENT' ... ', 'TUNIT',
> 'XTENSION'}`
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
> **load\_SToF\_fits(scan\_desc: dict = None,**kwargs)\*\*
> 
> **load\_SToF\_hdf5(scan\_desc: dict = None,**kwargs)\*\*
> 
> > Imports a FITS file that contains ToF spectra.
> > 
> >   - Parameters  
> >     **scan\_desc** – Dictionary with extra information to attach to
> >     the xr.Dataset, must contain the location
> > 
> > of the file :return: xr.Dataset
