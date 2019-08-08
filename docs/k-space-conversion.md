# Converting ARPES Data to Momentum-Space

## Converting Volumetric Data

PyARPES provides a consistent interface for converting ARPES data from angle to momentum space.
This means that there is only a single function that provides an entrypoint for converting 
volumetric data: `arpes.utilities.conversion.convert_to_kspace`.

Using the PyARPES data plugins, you can be confident that your data 
will convert to momentum immediately after you load it. 
You can set the symmetry point by using `{coord_name}_offset` attributes:

```python
from arpes.io import load_example_data
from arpes.utilities.conversion import convert_to_kspace

f = load_example_data().spectrum

f.attrs['phi_offset'] = 0.3
convert_to_kspace(f).plot()
```

There is also an interactive tool for setting the offset

```python
from arpes.io import load_example_data
from arpes.widgets import kspace_tool

f = load_example_data().spectrum
ctx = kspace_tool(f)
```

For photon energy dependence scans, you can also set the attribute `inner_potential`, and the 
`work_function` can be set for all scans as well.

## Correcting the chemical potential before converting

We can also first correct for the slit distortion to the chemical potential for full map, 
before converting an entire Fermi surface.

This method shows an alternative way of setting the coordinate offsets, which is to label
normal emission by setting the `G` point high symmetry location in angle-space.  

![Converting a Fermi Surface](static/kxky-conversion.png)

### Requesting a Resolution

PyARPES attempts to pick resolutions in the destination coordinate space that match the gridding
in the original space. You can override them however, by passing a resolution for any of the 
destination coordinates which will be used when creating the grid to interpolate onto. 

If you use this technique, consider also binning your data to take advantage of higher SNR, because 
only nearest neighbors are used for interpolating.

![Converting a cut with desired resolution](static/conversion-resolution.png)

## Converting Coordinates

Whereas to convert volumetric data to momentum space, we create a uniform grid in the destination
coordinate (momentum) space and interpolate the value on this grid by backwards converting, 
the coordinates can be converted forwards directly for one dimensional data. This is particularly 
useful for plotting the actual k<sub>z</sub> momentum dispersion across a cut, or plotting the segment
seen in an experiment or by the spectrometer.

As an example, we can use the forward coordinate conversion to plot the accessible parallel momentum 
values as a function of the polar angle from normal emission.

![Momentum reach available in ARPES](static/momentum-reach.png)

**Note:** Because of the intrinsic differences between converting volumetric and coordinate (1D) data,
you should tend to use `convert_coordinates_to_kspace_forward` when plotting orientations and cuts
