# arpes.xarray\_extensions module

**class arpes.xarray\_extensions.ARPESDataArrayAccessor(xarray\_obj)**

> Bases: `arpes.xarray_extensions.ARPESAccessorBase`
> 
> **cut\_dispersion\_plot(pattern='{}.png',**kwargs)\*\*
> 
> **dispersion\_plot(pattern='{}.png',**kwargs)\*\*
> 
> **fermi\_edge\_reference\_plot(pattern='{}.png',**kwargs)\*\*
> 
> **fs\_plot(pattern='{}.png',**kwargs)\*\*
> 
> **isosurface\_plot(pattern='{}.png',**kwargs)\*\*
> 
> **reference\_plot(**kwargs)\*\*
> 
> **show(**kwargs)\*\*
> 
> **show\_band\_tool(**kwargs)\*\*
> 
> **show\_d2(**kwargs)\*\*
> 
> **subtraction\_reference\_plot(pattern='{}.png',**kwargs)\*\*

**class arpes.xarray\_extensions.ARPESDatasetAccessor(xarray\_obj)**

> Bases: `arpes.xarray_extensions.ARPESAccessorBase`
> 
> `degrees_of_freedom`
> 
> **make\_spectrum\_reference\_plots(prefix='',**kwargs)\*\*
> 
> > Photocurrent normalized + unnormalized figures, in particular:
> > 
> > 1.  The reference plots for the photocurrent normalized spectrum
> > 
> > 2.    - The normalized total cycle intensity over scan DoF, i.e.  
> >         cycle vs scan DOF integrated over E, phi
> > 
> > 3.  For delay scans:
> 
> **polarization\_plot(**kwargs)\*\*
> 
> **reference\_plot(**kwargs)\*\*
> 
> > A bit of a misnomer because this actually makes many plots. For full
> > datasets, the relevant components are:
> > 
> > 1.  Temperature as function of scan DOF
> > 
> > 2.  Photocurrent as a function of scan DOF
> > 
> > 3.    - Photocurrent normalized + unnormalized figures, in  
> >         particular i. The reference plots for the photocurrent
> >         normalized spectrum ii. The normalized total cycle intensity
> >         over scan DoF, i.e. cycle vs scan DOF integrated over E, phi
> >         3.  For delay scans:
> > 
> > <!-- end list -->
> > 
> >   - Parameters  
> >     **kwargs** –
> > 
> >   - Returns
> 
> `scan_degrees_of_freedom`
> 
> `spectrum`
> 
> `spectrum_degrees_of_freedom`
> 
> `spectrum_type`

**class arpes.xarray\_extensions.ARPESFitToolsAccessor(xarray\_obj:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Bases: `object`
> 
> `band_names`
> 
> `bands`
> 
> > This should probably instantiate appropriate types :return:
> 
> **p(param\_name)**
> 
> `parameter_names`
> 
> **s(param\_name)**
> 
> **show()**
> 
> **show\_fit\_diagnostic()**
> 
> > alias for `.show` :return:
