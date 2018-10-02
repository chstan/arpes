# arpes.endstations.fits\_utils module

**arpes.endstations.fits\_utils.extract\_coords(attrs,
dimension\_renamings=None)**

> Does the hard work of extracting coordinates from the scan
> description. :param attrs: :param dimension\_renamings: :return:

**arpes.endstations.fits\_utils.find\_clean\_coords(hdu, attrs,
spectra=None, mode='ToF', dimension\_renamings=None)**

> Determines the scan degrees of freedom, the shape of the actual
> “spectrum” and reads and parses the coordinates from the header
> information in the recorded scan.
> 
> TODO Write data loading tests to ensure we don’t break MC
> compatibility
> 
>   - Parameters
>     
>       - **spectra** –
>     
>       - **hdu** –
>     
>       - **attrs** –
>     
>       -   - **mode** – Available modes are “ToF”, “MC”. This
>             customizes  
>             the read process
>     
>       - **dimension\_renamings** –
> 
>   - Returns  
>     (coordinates, dimensions, np shape of actual spectrum)
