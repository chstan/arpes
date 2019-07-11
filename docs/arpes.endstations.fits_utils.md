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
> Note: because different scan configurations can have different values
> of the detector coordinates, such as for instance when you record in
> two different angular modes of the spectrometer or when you record XPS
> spectra at different binding energies, we need to be able to provide
> separate coordinates for each of the different scans.
> 
> In the case where there is a unique coordinate, we will return only
> that coordinate, under the anticipated name, such as ‘eV’. Otherwise,
> we will return the coordinates that different between the scan
> configurations under the spectrum name, and with unique names, such as
> ‘eV-Swept\_Spectra0’.
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
