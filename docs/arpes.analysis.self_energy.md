arpes.analysis.self\_energy module
==================================

**arpes.analysis.self\_energy.estimate\_bare\_band(dispersion:
xarray.core.dataarray.DataArray, bare\_band\_specification:
Optional\[str\] = None)**

> Estimates the bare band from a fitted dispersion. This can be done in
> a few ways:
>
> 1.  None: Equivalent to ‘baseline\_linear’ below
>
> 2.  ‘linear’: A linear fit to the dispersion is used, and this also  
>     provides the fermi\_velocity
>
> 3.  ‘ransac\_linear’: A linear fit with random sample consensus  
>     (RANSAC) region will be used and this also provides the
>     fermi\_velocity
>
> 4.  ‘hough’: Hough transform based method
>
> Parameters  
> -   **dispersion** –
> -   **bare\_band\_specification** –
>
> Returns  

**arpes.analysis.self\_energy.fit\_for\_self\_energy(data:
xarray.core.dataarray.DataArray, method='mdc', bare\_band:
Optional\[Union\[xarray.core.dataarray.DataArray, str,
lmfit.model.ModelResult\]\] = None,**kwargs) -&gt;
xarray.core.dataset.Dataset\*\*

> Fits for the self energy of a dataset containing a single band.
>
> The bare band shape :param data: :param method: one of ‘mdc’ and ‘edc’
> :param bare\_band: :return:

**arpes.analysis.self\_energy.quasiparticle\_lifetime(self\_energy:
xarray.core.dataarray.DataArray, bare\_band:
xarray.core.dataarray.DataArray) -&gt; xarray.core.dataarray.DataArray**

> Calculates the quasiparticle mean free path in meters (meters!). The
> bare band is used to calculate the band/Fermi velocity and internally
> the procedure to calculate the quasiparticle lifetime is used
>
> Parameters  
> -   **self\_energy** –
> -   **bare\_band** –
>
> Returns  

**arpes.analysis.self\_energy.to\_self\_energy(dispersion:
xarray.core.dataarray.DataArray, bare\_band:
Optional\[Union\[xarray.core.dataarray.DataArray, str,
lmfit.model.ModelResult\]\] = None, k\_independent=True,
fermi\_velocity=None) -&gt; xarray.core.dataset.Dataset**

> Converts MDC fit results into the self energy. This largely consists
> of extracting out the linewidth and the difference between the
> dispersion and the bare band value.
>
> lorentzian(x, amplitude, center, sigma) =  
> (amplitude / pi) \* sigma/(sigma^2 + ((x-center))\*\*2)
>
> Once we have the curve-fitted dispersion we can calculate the self
> energy if we also know the bare-band dispersion. If the bare band is
> not known, then at least the imaginary part of the self energy is
> still calculable, and a guess as to the real part can be calculated
> under assumptions of the bare band dispersion as being free electron
> like wih effective mass m\* or being Dirac like (these are equivalent
> at low enough energy).
>
> Acceptabe bare band spefications are discussed in detail in
> *estimate\_bare\_band* above.
>
> To future readers of the code, please note that the half-width
> half-max of a Lorentzian is equal to the $gamma$ parameter, which
> defines the imaginary part of the self energy.
>
> Parameters  
> -   **dispersion** –
> -   **bare\_band** –
> -   **k\_independent** –
> -   **fermi\_velocity** –
>
> Returns  
