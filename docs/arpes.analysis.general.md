# arpes.analysis.general module

**arpes.analysis.general.normalize\_by\_fermi\_distribution(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
max\_gain=None, rigid\_shift=0, instrumental\_broadening=None,
total\_broadening=None)**

> Normalizes a scan by 1/the fermi dirac distribution. You can control
> the maximum gain with `clamp`, and whether the Fermi edge needs to be
> shifted (this is for those desperate situations where you want
> something that “just works”) via `rigid_shift`.
> 
>   - Parameters
>     
>       - **data** – Input
>     
>       -   - **max\_gain** – Maximum value for the gain. By default
>             the  
>             value used is the mean of the spectrum.
>     
>       -   - **rigid\_shift** – How much to shift the spectrum
>             chemical  
>             potential.
> 
> Pass the nominal value for the chemical potential in the scan. I.e. if
> the chemical potential is at BE=0.1, pass rigid\_shift=0.1. :param
> instrumental\_broadening: Instrumental broadening to use for
> convolving the distribution :return: Normalized DataArray

**arpes.analysis.general.symmetrize\_axis(data, axis\_name,
flip\_axes=None, shift\_axis=True)**

> Symmetrizes data across an axis. It would be better ultimately to be
> able to implement an arbitrary symmetry (such as a mirror or
> rotational symmetry about a line or point) and to symmetrize data by
> that method.
> 
>   - Parameters
>     
>       - **data** –
>       - **axis\_name** –
>       - **flip\_axes** –
>       - **shift\_axis** –
> 
>   - Returns

**arpes.analysis.general.condense(data:
xarray.core.dataarray.DataArray)**

> Clips the data so that only regions where there is substantial weight
> are included. In practice this usually means selecting along the `eV`
> axis, although other selections might be made.
> 
>   - Parameters  
>     **data** – xarray.DataArray
> 
>   - Returns

**arpes.analysis.general.rebin(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
shape: dict = None, reduction: Union\[int, dict\] = None,
interpolate=False,**kwargs)\*\*

> Rebins the data onto a different (smaller) shape. By default the
> behavior is to split the data into chunks that are integrated over. An
> interpolation option is also available.
> 
> Exactly one of `shape` and `reduction` should be supplied.
> 
> Dimensions corresponding to missing entries in `shape` or `reduction`
> will not be changed.
> 
>   - Parameters
>     
>       - **data** –
>       - **interpolate** – Use interpolation instead of integration
>       - **shape** – Target shape
>       - **reduction** – Factor to reduce each dimension by
> 
>   - Returns

**arpes.analysis.general.fit\_fermi\_edge(data, energy\_range=None)**

> Fits a Fermi edge. Not much easier than doing it manually, but this
> can be useful sometimes inside procedures where you don’t want to
> reimplement this logic. :param data: :param energy\_range: :return:
