# arpes.plotting.bz module

**arpes.plotting.bz.annotate\_special\_paths(ax, paths, cell=None,
offset=None, special\_points=None, labels=None,**kwargs)\*\*

**arpes.plotting.bz.bz2d\_plot(cell, vectors=False, paths=None,
points=None, repeat=None, ax=None, offset=None,
hide\_ax=True,**kwargs)\*\*

> This piece of code modified from ase.ase.dft.bz.py:bz2d\_plot and
> follows copyright and license for ASE.
> 
> Plots a Brillouin zone corresponding to a given unit cell :param cell:
> :param vectors: :param paths: :param points: :return:

**arpes.plotting.bz.bz3d\_plot(cell, vectors=False, paths=None,
points=None, ax=None, elev=None, scale=1, repeat=None, offset=None,
hide\_ax=True,**kwargs)\*\*

> For now this is lifted from ase.dft.bz.bz3d\_plot with some
> modifications. All copyright and licensing terms for this and
> bz2d\_plot are those of the current release of ASE (Atomic Simulation
> Environment).
> 
>   - Parameters
>     
>       - **cell** –
>       - **vectors** –
>       - **paths** –
>       - **points** –
>       - **elev** –
>       - **scale** –
> 
>   - Returns

**arpes.plotting.bz.bz\_plot(cell, \*args,**kwargs)\*\*

**arpes.plotting.bz.plot\_data\_to\_bz(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
cell,**kwargs)\*\*

**arpes.plotting.bz.plot\_data\_to\_bz2d(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
cell, rotate=None, shift=None, scale=None, ax=None, mask=True, out=None,
bz\_number=None,**kwargs)\*\*

**arpes.plotting.bz.plot\_data\_to\_bz3d(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
cell,**kwargs)\*\*

**arpes.plotting.bz.plot\_plane\_to\_bz(cell, plane, ax,
special\_points=None, facecolor='red')**
