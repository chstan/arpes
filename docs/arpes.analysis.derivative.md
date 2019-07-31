# arpes.analysis.derivative module

**arpes.analysis.derivative.curvature(arr:
xarray.core.dataarray.DataArray, directions=None, alpha=1, beta=None)**

>   - Defined via  
>     C(x,y) = (\[C\_0 + (df/dx)^2\]d^2f/dy^2 - 2 \* df/dx df/dy
>     d^2f/dxdy + \[C\_0 + (df/dy)^2\]d^2f/dx^2) / (C\_0 (df/dx)^2 +
>     (df/dy)^2)^(3/2)
> 
> of in the case of inequivalent dimensions x and y
> 
>   - Parameters
>     
>       - **arr** –
>     
>       -   - **alpha** – regulation parameter, chosen
>             semi-universally,  
>             but with no particular justification
> 
>   - Returns

**arpes.analysis.derivative.dn\_along\_axis(arr:
xarray.core.dataarray.DataArray, axis=None, smooth\_fn=None, order=2)**

> Like curvature, performs a second derivative. You can pass a function
> to use for smoothing through the parameter smooth\_fn, otherwise no
> smoothing will be performed.
> 
> You can specify the axis to take the derivative along with the axis
> param, which expects a string. If no axis is provided the axis will be
> chosen from among the available ones according to the preference for
> axes here, the first available being taken:
> 
> \[‘eV’, ‘kp’, ‘kx’, ‘kz’, ‘ky’, ‘phi’, ‘beta’, ‘theta\] :param arr:
> :param axis: :param smooth\_fn: :param order: Specifies how many
> derivatives to take :return:

**arpes.analysis.derivative.d2\_along\_axis(arr:
xarray.core.dataarray.DataArray, axis=None, smooth\_fn=None, \*,
order=2)**

> Like curvature, performs a second derivative. You can pass a function
> to use for smoothing through the parameter smooth\_fn, otherwise no
> smoothing will be performed.
> 
> You can specify the axis to take the derivative along with the axis
> param, which expects a string. If no axis is provided the axis will be
> chosen from among the available ones according to the preference for
> axes here, the first available being taken:
> 
> \[‘eV’, ‘kp’, ‘kx’, ‘kz’, ‘ky’, ‘phi’, ‘beta’, ‘theta\] :param arr:
> :param axis: :param smooth\_fn: :param order: Specifies how many
> derivatives to take :return:

**arpes.analysis.derivative.d1\_along\_axis(arr:
xarray.core.dataarray.DataArray, axis=None, smooth\_fn=None, \*,
order=1)**

> Like curvature, performs a second derivative. You can pass a function
> to use for smoothing through the parameter smooth\_fn, otherwise no
> smoothing will be performed.
> 
> You can specify the axis to take the derivative along with the axis
> param, which expects a string. If no axis is provided the axis will be
> chosen from among the available ones according to the preference for
> axes here, the first available being taken:
> 
> \[‘eV’, ‘kp’, ‘kx’, ‘kz’, ‘ky’, ‘phi’, ‘beta’, ‘theta\] :param arr:
> :param axis: :param smooth\_fn: :param order: Specifies how many
> derivatives to take :return:

**arpes.analysis.derivative.minimum\_gradient(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
delta=1)**

**arpes.analysis.derivative.vector\_diff(arr, delta, n=1)**

> Computes finite differences along the vector delta, given as a tuple
> 
> Using delta = (0, 1) is equivalent to np.diff(…, axis=1), while using
> delta = (1, 0) is equivalent to np.diff(…, axis=0). :param arr:
> np.ndarray :param delta: iterable containing vector to take difference
> along :return:
