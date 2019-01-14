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

**class arpes.analysis.gap.AffineBroadenedFD(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> [arpes.fits.fit\_models.XModelMixin](arpes.fits.fit_models#arpes.fits.fit_models.XModelMixin)
> 
> A model for fitting an affine density of states with resolution
> broadened Fermi-Dirac occupation
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**arpes.analysis.gap.symmetrize(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**
