# arpes.analysis.gap module

**arpes.analysis.gap.normalize\_by\_fermi\_dirac(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
reference\_data: Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\] = None, plot=False, broadening=None,
temperature\_axis=None,
temp\_offset=0,**kwargs)\*\*

**arpes.analysis.gap.determine\_broadened\_fermi\_distribution(reference\_data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
fixed\_temperature=True)**

> Determine the parameters for broadening by temperature and
> instrumental resolution for a piece of data.
> 
> As a general rule, we first try to estimate the instrumental
> broadening and linewidth broadening according to calibrations provided
> for the beamline + instrument, as a starting point.
> 
> We also calculate the thermal broadening to expect, and fit an edge
> location. Then we use a Gaussian convolved Fermi-Dirac distribution
> against an affine density of states near the Fermi level, with a
> constant offset background above the Fermi level as a simple but
> effective model when away from lineshapes.
> 
> These parameters can be used to bootstrap a fit to actual data or used
> directly in `normalize_by_fermi_dirac`.
> 
>   - Parameters  
>     **reference\_data** –
> 
>   - Returns

**arpes.analysis.gap.symmetrize(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
subpixel=False, full\_spectrum=False)**
