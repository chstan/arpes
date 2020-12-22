arpes.xarray\_extensions module
===============================

This is another core part of PyARPES. It provides a lot of extensions to
what comes out of the box in xarray. Some of these are useful generics,
generally on the .T extension, others collect and manipulate metadata,
interface with plotting routines, provide functional programming
utilities, etc.

If *f* is an ARPES spectrum, then *f.S* should provide a nice
representation of your data in a Jupyter cell. This is a complement to
the text based approach that merely printing *f* offers.

**class arpes.xarray\_extensions.ARPESDataArrayAccessor(xarray\_obj)**

> Bases: `arpes.xarray_extensions.ARPESAccessorBase`
>
> **cut\_dispersion\_plot(pattern='{}.png',**kwargs)\*\*
>
> **cut\_nan\_coords()**
>
> **dispersion\_plot(pattern='{}.png',**kwargs)\*\*
>
> **property eV**
>
> **fermi\_edge\_reference\_plot(pattern='{}.png',**kwargs)\*\*
>
> **fs\_plot(pattern='{}.png',**kwargs)\*\*
>
> **isosurface\_plot(pattern='{}.png',**kwargs)\*\*
>
> **nan\_to\_num(x=0)**
>
> > xarray version of numpy.nan\_to\_num :param x: :return:
>
> **plot(\*args, rasterized=True,**kwargs)\*\*
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
> **property degrees\_of\_freedom**
>
> **property is\_multi\_region**
>
> **property is\_spatial**
>
> > Infers whether a given scan has real-space dimensions and
> > corresponds to SPEM or u/nARPES. :return:
>
> **make\_spectrum\_reference\_plots(prefix='',**kwargs)\*\*
>
> > Photocurrent normalized + unnormalized figures, in particular:
> >
> > 1.  The reference plots for the photocurrent normalized spectrum
> >
> > 2.  The normalized total cycle intensity over scan DoF, i.e.  
> >     cycle vs scan DOF integrated over E, phi
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
> > 3.  Photocurrent normalized + unnormalized figures, in  
> >     particular i. The reference plots for the photocurrent
> >     normalized spectrum ii. The normalized total cycle intensity
> >     over scan DoF, i.e. cycle vs scan DOF integrated over E, phi
> >
> >     1.  For delay scans:
> >
> > 4.  For spatial scans: i. energy/angle integrated spatial maps  
> >     with subsequent measurements indicated 2. energy/angle
> >     integrated FS spatial maps with subsequent measurements
> >     indicated
> >
> > Parameters  
> > **kwargs** –
> >
> > Returns  
>
> **property scan\_degrees\_of\_freedom**
>
> **property spectra**
>
> **property spectrum**
>
> **property spectrum\_degrees\_of\_freedom**
>
> **property spectrum\_type**

**class arpes.xarray\_extensions.ARPESFitToolsAccessor(xarray\_obj:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

> Bases: `object`
>
> **property band\_names**
>
> **property bands**
>
> > This should probably instantiate appropriate types :return:
>
> **p(param\_name)**
>
> **param\_as\_dataset(param\_name)**
>
> **property parameter\_names**
>
> **plot\_param(param\_name,**kwargs)\*\*
>
> **s(param\_name)**
>
> **show()**
>
> **show\_fit\_diagnostic()**
>
> > alias for `.show` :return:
