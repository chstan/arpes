# arpes.analysis.shirley module

**arpes.analysis.shirley.calculate\_shirley\_background(xps:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
energy\_range: slice = None, eps=1e-07, max\_iters=50, n\_samples=5)**

> Calculates a shirley background iteratively over the full energy range
> *energy\_range*. :param xps: :param energy\_range: :param eps: :param
> max\_iters:
:return:

**arpes.analysis.shirley.calculate\_shirley\_background\_full\_range(xps:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
eps=1e-07, max\_iters=50, n\_samples=5)**

> Calculates a shirley background in the range of *energy\_slice*
> according to:
> 
> S(E) = I(E\_right) + k \* (A\_right(E)) / (A\_left(E) + A\_right(E))
> 
> Typically
> 
> k := I(E\_right) - I(E\_left)
> 
> The iterative method is continued so long as the total background is
> not converged to relative error *eps*.
> 
> The method continues for a maximum number of iterations *max\_iters*.
> 
> In practice, what we can do is to calculate the cumulative sum of the
> data along the energy axis of both the data and the current estimate
> of the background :param xps: :param eps: :return:

**arpes.analysis.shirley.remove\_shirley\_background(xps:
Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\],**kwargs)\*\*

> Calculates and removes a Shirley background from a spectrum. Only the
> background corrected spectrum is retrieved. :param xps: :param kwargs:
> :return:
