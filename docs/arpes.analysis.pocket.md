# arpes.analysis.pocket module

**arpes.analysis.pocket.curves\_along\_pocket(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
n\_points=None, inner\_radius=0, outer\_radius=5,
shape=None,**kwargs)\*\*

> Produces radial slices along a Fermi surface through a pocket. Evenly
> distributes perpendicular cuts along an ellipsoid. The major axes of
> the ellipsoid can be specified by *shape* but must be axis aligned.
> 
> The inner and outer radius parameters control the endpoints of the
> resultant slices along the Fermi surface :param data: :param
> n\_points: :param inner\_radius: :param outer\_radius: :param shape:
> :param kwargs: :return:

**arpes.analysis.pocket.edcs\_along\_pocket(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
kf\_method=None, select\_radius=None, sel=None,
method\_kwargs=None,**kwargs)\*\*

> Collects EDCs around a pocket. This consists first in identifying the
> momenta around the pocket, and then integrating small windows around
> each of these points.
> 
>   - Parameters
>     
>       - **data** –
>       - **kf\_method** –
>       - **select\_radius** –
>       - **sel** –
>       - **method\_kwargs** –
>       - **kwargs** –
> 
>   - Returns

**arpes.analysis.pocket.radial\_edcs\_along\_pocket(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
angle, inner\_radius=0, outer\_radius=5, n\_points=None,
select\_radius=None,**kwargs)\*\*

> Produces EDCs distributed radially along a vector from the pocket
> center. The pocket center should be passed through kwargs via
> *{dim}={value}*. I.e. an appropriate call would be
> 
> radial\_edcs\_along\_pocket(spectrum, np.pi / 4, inner\_radius=1,
> outer\_radius=4, phi=0.1, beta=0)
> 
>   - Parameters
>     
>       - **data** – ARPES Spectrum
>       - **angle** – Angle along the FS to cut against
>       - **n\_points** – Number of EDCs, can be automatically inferred
>       - **select\_radius** –
>       - **kwargs** –
> 
>   - Returns

**arpes.analysis.pocket.pocket\_parameters(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
kf\_method=None, sel=None, method\_kwargs=None,**kwargs)\*\*

> Estimates pocket center, anisotropy, principal vectors, and extent in
> either angle or k-space. Since data can be converted forward it is
> generally advised to do this analysis in angle space before
> conversion. :param data: :param kf\_method: :param sel: :param
> method\_kwargs: :param kwargs: :return:
