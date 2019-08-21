# arpes.utilities.conversion.bounds\_calculations module

**arpes.utilities.conversion.bounds\_calculations.calculate\_kp\_kz\_bounds(arr:
xarray.core.dataarray.DataArray)**

**arpes.utilities.conversion.bounds\_calculations.calculate\_kx\_ky\_bounds(arr:
xarray.core.dataarray.DataArray)**

> Calculates the kx and ky range for a dataset with a fixed photon
> energy
> 
> This is used to infer the gridding that should be used for a k-space
> conversion. Based on Jonathan Denlinger’s old codes :param arr:
> Dataset that includes a key indicating the photon energy of the scan
> :return: ((kx\_low, kx\_high,), (ky\_low,
ky\_high,))

**arpes.utilities.conversion.bounds\_calculations.calculate\_kp\_bounds(arr:
xarray.core.dataarray.DataArray)**

**arpes.utilities.conversion.bounds\_calculations.full\_angles\_to\_k(kinetic\_energy,
phi, psi, alpha, beta, theta, chi, inner\_potential,
approximate=False)**

> Converts from the full set of standard PyARPES angles to momentum.
> More details on angle to momentum conversion can be found at [the
> momentum conversion
> notes](https://arpes.netlify.com/#/momentum-conversion).
> 
> Because the inverse coordinate transforms in PyARPES use the small
> angle approximation, we also allow the small angle approximation in
> the forward direction, using the *approximate=* keyword argument.
> :param kinetic\_energy: :param phi: :param psi: :param alpha: :param
> beta: :param theta: :param chi: :param inner\_potential: :param
> approximate:
:return:

**arpes.utilities.conversion.bounds\_calculations.full\_angles\_to\_k\_approx(kinetic\_energy,
phi, psi, alpha, beta, theta, chi, inner\_potential)**

> Small angle approximation of the momentum conversion functions.
> Depending on the value of alpha, which we do not small angle
> approximate, this takes a few different forms.
> 
>   - Parameters
>     
>       - **kinetic\_energy** –
>       - **phi** –
>       - **psi** –
>       - **alpha** –
>       - **beta** –
>       - **theta** –
>       - **chi** –
>       - **inner\_potential** –
> 
>   - Returns
