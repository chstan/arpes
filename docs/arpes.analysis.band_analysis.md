# arpes.analysis.band\_analysis module

**arpes.analysis.band\_analysis.fit\_bands(arr:
xarray.core.dataarray.DataArray, band\_description, background=None,
direction='mdc', preferred\_k\_direction=None, step=None)**

> Fits bands and determines dispersion in some region of a spectrum
> :param arr: :param band\_description: A description of the bands to
> fit in the region :param background: :param direction: :return:

**arpes.analysis.band\_analysis.fit\_for\_effective\_mass(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
fit\_kwargs=None)**

> Performs an effective mass fit by first fitting for Lorentzian
> lineshapes and then fitting a quadratic model to the result. This is
> an alternative to global effective mass fitting.
> 
> In the case that data is provided in anglespace, the Lorentzian fits
> are performed in anglespace before being converted to momentum where
> the effective mass is extracted.
> 
> We should probably include uncertainties here.
> 
>   - Parameters
>     
>       - **data** –
>     
>       -   - **fit\_kwargs** – Passthrough for arguments to  
>             *broadcast\_model*,
> 
> used internally to obtain the Lorentzian peak locations :return:
