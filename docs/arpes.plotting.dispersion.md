# arpes.plotting.dispersion module

**arpes.plotting.dispersion.plot\_dispersion(spectrum:
xarray.core.dataarray.DataArray, bands, out=None)**

**arpes.plotting.dispersion.labeled\_fermi\_surface(data, title=None,
ax=None, hold=False, include\_symmetry\_points=True, include\_bz=True,
out=None, fermi\_energy=0,**kwargs)\*\*

**arpes.plotting.dispersion.cut\_dispersion\_plot(data:
xarray.core.dataarray.DataArray, e\_floor=None, title=None, ax=None,
include\_symmetry\_points=True, out=None, quality='high',**kwargs)\*\*

> Makes a 3D cut dispersion plot. At the moment this only supports
> rectangular BZs. :param data: :param e\_floor: :param title: :param
> ax: :param out: :param kwargs: :return:

**arpes.plotting.dispersion.fancy\_dispersion(data, title=None, ax=None,
out=None, include\_symmetry\_points=True, norm=None,**kwargs)\*\*

**arpes.plotting.dispersion.reference\_scan\_fermi\_surface(data,
out=None,**kwargs)\*\*

**arpes.plotting.dispersion.hv\_reference\_scan(data, out=None,
e\_cut=-0.05, bkg\_subtraction=0.8,**kwargs)\*\*

**arpes.plotting.dispersion.scan\_var\_reference\_plot(data, title=None,
ax=None, norm=None, out=None,**kwargs)\*\*
