# arpes.analysis.gap module

**arpes.analysis.gap.normalize\_by\_fermi\_dirac(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
reference\_data: Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\] = None, plot=False, broadening=None,
temperature\_axis=None, temp\_offset=0,**kwargs)\*\*

> Normalizes data according to a Fermi level reference on separate data
> or using the same source spectrum.
> 
> To do this, a linear density of states is multiplied against a
> resolution broadened Fermi-Dirac distribution
> (*arpes.fits.fit\_models.AffineBroadenedFD*). We then set the density
> of states to 1 and evaluate this model to obtain a reference that the
> desired spectrum is normalized by.
> 
>   - Parameters
>     
>       - **data** – Data to be normalized.
>     
>       -   - **reference\_data** – A reference spectrum, typically a
>             metal  
>             reference. If not provided the integrated data is used.
>             Beware: this is inappropriate if your data is gapped.
>     
>       -   - **plot** – A debug flag, allowing you to view the  
>             normalization spectrum and relevant curve-fits.
>     
>       - **broadening** – Detector broadening.
>     
>       -   - **temperature\_axis** – Temperature coordinate, used to
>             adjust  
>             the quality of the reference for temperature dependent
>             data.
>     
>       -   - **temp\_offset** – Temperature calibration in the case of
>             low  
>             temperature data. Useful if the temperature at the sample
>             is known to be hotter than the value recorded off of a
>             diode.
>     
>       - **kwargs**
–
> 
>   - Returns

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

> Symmetrizes data across the chemical potential. This provides a crude
> tool by which gap analysis can be performed. In this implementation,
> subpixel accuracy is achieved by interpolating data.
> 
>   - Parameters
>     
>       - **data** – Input array.
>     
>       - **subpixel** – Enable subpixel correction
>     
>       -   - **full\_spectrum** – Returns data above and below the
>             chemical  
>             potential. By default, only the bound part of the spectrum
>             (below the chemical potential) is returned, because the
>             other half is identical.
> 
>   - Returns
