# arpes.analysis.kfermi module

**arpes.analysis.kfermi.kfermi\_from\_mdcs(mdc\_results:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
param=None)**

> Calculates a Fermi momentum using a series of MDCs and the known Fermi
> level (eV=0). This is especially useful to isolate an area for
> analysis.
> 
> This method tolerates data that came from a prefixed model fit, but
> will always attempt to look for an attribute containing “center”.
> 
>   - Parameters
>     
>       -   - **mdc\_results** – A DataArray or Dataset containing  
>             \>\>\`\<\<lmfit.ModelResult\`s.
>     
>       - **param** –
> 
>   - Returns
