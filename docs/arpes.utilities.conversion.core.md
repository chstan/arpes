# arpes.utilities.conversion.core module

Helper functions for coordinate transformations. All the functions here
assume standard polar angles, as given in the [data model
documentation](https://arpes.netlify.com/#/spectra).

Functions here must accept constants or numpy arrays as valid inputs, so
all standard math functions have been replaced by their equivalents out
of numpy. Array broadcasting should handle any issues or weirdnesses
that would encourage the use of direct iteration, but in case you need
to write a conversion directly, be aware that any functions here must
work on arrays as well for consistency with client code.

Everywhere:

Kinetic energy -\> ‘kinetic\_energy’ Binding energy -\> ‘eV’, for
convenience (negative below 0) Photon energy -\> ‘hv’

Better facilities should be added for ToFs to do simultaneous (timing,
angle) to (binding energy, k-space).

**arpes.utilities.conversion.core.convert\_to\_kspace(arr:
xarray.core.dataarray.DataArray, forward=False,
resolution=None,**kwargs)\*\*

> “Forward” or “backward” converts the data to momentum space.
> 
> “Backward” The standard method. Works in generality by regridding the
> data into the new coordinate space and then interpolating back into
> the original data.
> 
> “Forward” By converting the coordinates, rather than by interpolating
> the data. As a result, the data will be totally unchanged by the
> conversion (if we do not apply a Jacobian correction), but the
> coordinates will no longer have equal spacing.
> 
> This is only really useful for zero and one dimensional data because
> for two dimensional data, the coordinates must become two dimensional
> in order to fully specify every data point (this is true in
> generality, in 3D the coordinates must become 3D as well).
> 
> The only exception to this is if the extra axes do not need to be
> k-space converted. As is the case where one of the dimensions is
> *cycle* or *delay*, for instance.
> 
>   - Parameters  
>     **arr** –
> 
>   - Returns

**arpes.utilities.conversion.core.slice\_along\_path(arr:
xarray.core.dataarray.DataArray, interpolation\_points=None,
axis\_name=None, resolution=None, shift\_gamma=True,
extend\_to\_edge=False,**kwargs)\*\*

> TODO: There might be a little bug here where the last coordinate has a
> value of 0, causing the interpolation to loop back to the start point.
> For now I will just deal with this in client code where I see it until
> I understand if it is universal.
> 
> Interpolates along a path through a volume. If the volume is higher
> dimensional than the desired path, the interpolation is broadcasted
> along the free dimensions. This allows one to specify a k-space path
> and receive the band structure along this path in k-space.
> 
> Points can either by specified by coordinates, or by reference to
> symmetry points, should they exist in the source array. These symmetry
> points are translated to regular coordinates immediately, but are
> provided as a convenience. If not all points specify the same set of
> coordinates, an attempt will be made to unify the coordinates. As an
> example, if the specified path is (kx=0, ky=0, T=20) -\> (kx=1, ky=1),
> the path will be made between (kx=0, ky=0, T=20) -\> (kx=1, ky=1,
> T=20). On the other hand, the path (kx=0, ky=0, T=20) -\> (kx=1, ky=1,
> T=40) -\> (kx=0, ky=1) will result in an error because there is no way
> to break the ambiguity on the temperature for the last coordinate.
> 
> A reasonable value will be chosen for the resolution, near the maximum
> resolution of any of the interpolated axes by default.
> 
> This function transparently handles the entire path. An alternate
> approach would be to convert each segment separately and concatenate
> the interpolated axis with xarray.
> 
> If the sentinel value ‘G’ for the Gamma point is included in the
> interpolation points, the coordinate axis of the interpolated
> coordinate will be shifted so that its value at the Gamma point is 0.
> You can opt out of this with the parameter ‘shift\_gamma’
> 
>   - Parameters
>     
>       - **arr** – Source data
>     
>       - **interpolation\_points** – Path vertices
>     
>       -   - **axis\_name** – Label for the interpolated axis. Under  
>             special circumstances a reasonable name will be chosen,
> 
> such as when the interpolation dimensions are kx and ky: in this case
> the interpolated dimension will be labeled kp. In mixed or ambiguous
> situations the axis will be labeled by the default value ‘inter’.
> :param resolution: Requested resolution along the interpolated axis.
> :param shift\_gamma: Controls whether the interpolated axis is shifted
> to a value of 0 at Gamma. :param extend\_to\_edge: Controls whether or
> not to scale the vector S - G for symmetry point S so that you
> interpolate to the edge of the available data :param kwargs: :return:
> xr.DataArray containing the interpolated data.
