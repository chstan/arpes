# arpes.corrections.fermi\_edge\_corrections module

**arpes.corrections.fermi\_edge\_corrections.install\_fermi\_edge\_reference(arr:
xarray.core.dataarray.DataArray)**

**arpes.corrections.fermi\_edge\_corrections.build\_quadratic\_fermi\_edge\_correction(arr:
xarray.core.dataarray.DataArray, fit\_limit=0.001, eV\_slice=None,
plot=False) -\>
lmfit.model.ModelResult**

**arpes.corrections.fermi\_edge\_corrections.build\_photon\_energy\_fermi\_edge\_correction(arr:
xarray.core.dataarray.DataArray, plot=False,
energy\_window=0.2)**

**arpes.corrections.fermi\_edge\_corrections.apply\_photon\_energy\_fermi\_edge\_correction(arr:
xarray.core.dataarray.DataArray,
correction=None,**kwargs)\*\*

**arpes.corrections.fermi\_edge\_corrections.apply\_quadratic\_fermi\_edge\_correction(arr:
xarray.core.dataarray.DataArray, correction: lmfit.model.ModelResult =
None,
offset=None)**

**arpes.corrections.fermi\_edge\_corrections.apply\_copper\_fermi\_edge\_correction(arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
copper\_ref: Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\],
\*args,**kwargs)\*\*

**arpes.corrections.fermi\_edge\_corrections.apply\_direct\_copper\_fermi\_edge\_correction(arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
copper\_ref: Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\], \*args,**kwargs)\*\*

> Applies a *direct* fermi edge correction. :param arr: :param
> copper\_ref: :param args: :param kwargs:
:return:

**arpes.corrections.fermi\_edge\_corrections.build\_direct\_fermi\_edge\_correction(arr:
xarray.core.dataarray.DataArray, fit\_limit=0.001, energy\_range=None,
plot=False, along='phi')**

> Builds a direct fermi edge correction stencil.
> 
> This means that fits are performed at each value of the ‘phi’
> coordinate to get a list of fits. Bad fits are thrown out to form a
> stencil.
> 
> This can be used to shift coordinates by the nearest value in the
> stencil.
> 
>   - Parameters
>     
>       - **copper\_ref** –
>       - **args** –
>       - **kwargs**
–
> 
>   - Returns

**arpes.corrections.fermi\_edge\_corrections.apply\_direct\_fermi\_edge\_correction(arr:
xarray.core.dataarray.DataArray, correction=None, \*args,**kwargs)\*\*
