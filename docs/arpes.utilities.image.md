# arpes.utilities.image module

**arpes.utilities.image.imread\_to\_xarray(str\_or\_path)**

> Like *imread*, except that this function wraps the result into a
> xr.DataArray that has x (pixel), y (pixel), and color (\[‘R’, ‘G’,
> ‘B’\]) dimensions.
> 
>   - Parameters  
>     **str\_or\_path** –
> 
>   - Returns

**arpes.utilities.image.imread(str\_or\_path)**

> A wrapper around *opencv.imread* and *imageio.imread* that
> 
> 1.  Falls back to the first available option on the system
> 2.  Normalizes OpenCV images to RGB format
> 3.  Removes the alpha channel from imageio.imread data
> 
> <!-- end list -->
> 
>   - Parameters  
>     **str\_or\_path** – pathlib.Path or str containing the image to be
>     read
> 
>   - Returns
