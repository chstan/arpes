# Spin-ARPES

Spin-ARPES datasets tend to look a bit different than spin-integrated
ARPES datasets. For one, they tend not to have as many dimensions as 
standard ARPES datasets, as many spin detectors only measure a single 
(k, E, S) point at a time, or might measure several components of **S** 
at once for single (k, E). Other detectors, like we one we have in the 
Lanzara lab, measure a single (k, S) point at once, but produce 
full EDCs.

Furthermore, SARPES datasets have the discrete spin label **S**. PyARPES 
choice of `xarray` as our data primitive makes it excellently suited for
handling SARPES data, because the different spin channels can exist 
simultaneously as `xr.DataArrays` on the same `xr.Dataset`.

## Spin-ARPES conventions in PyARPES

Most functionality in PyARPES works just as well for SARPES as for other 
modalities. You can load data just as you would with any other type:

![Loading SToF Data](/static/SToF-load.png)

Instead of having a `spectrum` data variable as we several: `up`, `down`, 
`t_up`, and `t_down`. As we can see `t_up` and `t_down` have `time` coordinates, 
while `up` and `down` have energy coordinates. This spin spectrum comes 
from a time of flight detector, so `t_up` and `t_down` are the raw timing spectra,
while `up` and `down` are the interpolated spectra after conversion to 
kinetic energy.

You can think of `up` and `down` in the same way that `spectrum` is treated for
spin integrated datasets. All analysis code that operates on `xr.DataArray`s should
work for these as well.  

**Note for plugin writers**: PyARPES reserves `t_up`, `t_down`, `up`, `down`
for the spin channels of ToF detectors and the spin channels of energy-aware 
detectors respectively. You should ensure plugins produce the appropriately
variables. Spin-integrated time of flight detectors should produce `t_spectrum`
and `spectrum`, as appropriate, **not** `e_spectrum`.

# Plotting Spin-EDCs

![Polarization Plot](static/simple-polarization-plot.png)

Plotting multidimensional spin data using false color for the 
different spin components, an intensity-saturation/polarization-hue scheme, 
or by quiver plot is also straightforward.  

# Converting Time-of-Flight Data to Kinetic Energy

PyARPES contains support for converting time of flight photoemission data to 
energy and momentum space. Our support for this currently supports the Lanzara Lab's
Spin-ARPES spectrometer, but support is largely generic to multidimensional (ARToF)
and PEEM-ToF detectors, given an associated [data loading plugin](/writing-plugins).

<!--- ![Converting time coordinates to energy](static/convert-time-to-energy.png) -->

Utilities can be found in `arpes.preparation.tof_preparation`.

**Note about units:** Be aware that the energy coordinates produced by conversion from ToF data 
are the electron kinetic energies. Because PyARPES is more concerned about axes having meaningful 
and consistent units, it does not distinguish between the binding and kinetic energy:
both are labelled by 'eV'. This has advantages, because analysis code will still work 
transparently on either type of data. You will apply a correction or offset to adjust units 
appropriately so that the chemical potential lies at zero binding energy.   