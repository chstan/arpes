# arpes.utilities package

## Subpackages

  -   - [arpes.utilities.conversion package](arpes.utilities.conversion)
        
          -   - [Submodules](arpes.utilities.conversion#submodules)
                
                  - [arpes.utilities.conversion.base
                    module](arpes.utilities.conversion.base)
                  - [arpes.utilities.conversion.bounds\_calculations
                    module](arpes.utilities.conversion.bounds_calculations)
                  - [arpes.utilities.conversion.core
                    module](arpes.utilities.conversion.core)
                  - [arpes.utilities.conversion.forward
                    module](arpes.utilities.conversion.forward)
                  - [arpes.utilities.conversion.kx\_ky\_conversion
                    module](arpes.utilities.conversion.kx_ky_conversion)
                  - [arpes.utilities.conversion.kz\_conversion
                    module](arpes.utilities.conversion.kz_conversion)
                  - [arpes.utilities.conversion.remap\_manipulator
                    module](arpes.utilities.conversion.remap_manipulator)
                  - [arpes.utilities.conversion.tof
                    module](arpes.utilities.conversion.tof)
        
          - [Module
            contents](arpes.utilities.conversion#module-arpes.utilities.conversion)

## Submodules

  - [arpes.utilities.attrs module](arpes.utilities.attrs)
  - [arpes.utilities.autoprep module](arpes.utilities.autoprep)
  - [arpes.utilities.bz module](arpes.utilities.bz)
  - [arpes.utilities.collections module](arpes.utilities.collections)
  - [arpes.utilities.dataset module](arpes.utilities.dataset)
  - [arpes.utilities.dict module](arpes.utilities.dict)
  - [arpes.utilities.funcutils module](arpes.utilities.funcutils)
  - [arpes.utilities.geometry module](arpes.utilities.geometry)
  - [arpes.utilities.jupyter\_utils
    module](arpes.utilities.jupyter_utils)
  - [arpes.utilities.math module](arpes.utilities.math)
  - [arpes.utilities.normalize module](arpes.utilities.normalize)
  - [arpes.utilities.region module](arpes.utilities.region)
  - [arpes.utilities.string module](arpes.utilities.string)
  - [arpes.utilities.xarray module](arpes.utilities.xarray)

## Module contents

Provides general utility methods that get used during the course of
analysis.

**arpes.utilities.arrange\_by\_indices(items, indices)**

> This function is best illustrated by the example below. It arranges
> the items in the input according to the new indices that each item
> should occupy.
> 
> It also has an inverse available in ‘unarrange\_by\_indices’.
> 
> Ex: arrange\_by\_indices(\[‘a’, ‘b’, ‘c’\], \[1, 2, 0\])

**arpes.utilities.enumerate\_dataarray(arr:
xarray.core.dataarray.DataArray)**

**arpes.utilities.fix\_burnt\_pixels(spectrum)**

> In reality the analyzers cannot provide perfect images for us. One of
> the principle failure modes is that individual pixels can get burnt
> out and will not provide any counts, or will provide consistently
> fewer or more than other pixels.
> 
> Our approach here is to look for peaks in the difference across pixels
> and frames of a spectrum as indication of issues to be fixed. To patch
> the pixels, we replace them with the average value of their neighbors.
> 
> spectrum - \<npArray\> containing the pixels
> 
> returns: \<npArray\> containing the fixed pixels

**arpes.utilities.jacobian\_correction(energies, lattice\_constant,
theta, beta, alpha, phis, rhat)**

> Because converting from angles to momenta does not preserve area, we
> need to multiply by the Jacobian of the transformation in order to get
> the appropriate number of counts in the new cells.
> 
> This differs across all the cells of a spectrum, because E and phi
> change. This function builds an array with the same shape that has the
> appropriate correction for each cell.
> 
> energies - \<npArray\> the linear sampling of energies across the
> spectrum phis - \<npArray\> the linear sampling of angles across the
> spectrum
> 
> returns: \<npArray\> a 2D array of the Jacobian correction to apply to
> each pixel in the spectrum

**arpes.utilities.rename\_datavar\_standard\_attrs(x)**

**arpes.utilities.rename\_standard\_attrs(x)**

**arpes.utilities.unarrange\_by\_indices(items, indices)**

> The inverse function to ‘arrange\_by\_indices’.
> 
> Ex: unarrange\_by\_indices(\[‘b’, ‘c’, ‘a’\], \[1, 2, 0\])

**arpes.utilities.unwrap\_attrs\_dict(attrs: \<module
'arpes.utilities.dict' from
'/Users/chstansbury/PycharmProjects/python-arpes/arpes/utilities/dict.py'\>)
-\> \<module 'arpes.utilities.dict' from
'/Users/chstansbury/PycharmProjects/python-arpes/arpes/utilities/dict.py'\>**

**arpes.utilities.walk\_scans(path, only\_id=False)**

**arpes.utilities.wrap\_attrs\_dict(attrs: \<module
'arpes.utilities.dict' from
'/Users/chstansbury/PycharmProjects/python-arpes/arpes/utilities/dict.py'\>,
original\_data: Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\] = None) -\> \<module
'arpes.utilities.dict' from
'/Users/chstansbury/PycharmProjects/python-arpes/arpes/utilities/dict.py'\>**
