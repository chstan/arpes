# Fermi Edge Corrections

## Adjusting for Monochromator Miscalibrations

Although this section discusses aligning the chemical potential at a 
synchrotron where monochromator miscalibration results in a changing 
apparent chemical potential as a function of photon energy, you can 
use the techniques in this section to handle all cases of aligning 
the Fermi level where each channel is independent.

![](static/hv-correction.png)

## Correcting for Hemisphere Slit Shape

If you use a straight slit when you collect ARPES with a hemisphere,
the apparent chemical potential will not form a straight line in the 
image plane, but instead will be curved. In many cases it is desirable 
not only to remove this effect, but also to remove any other miscalibration
in the chemical potential arising from alignment of the phosphor imaging
camera, or lens aberration.

In PyARPES, you can either apply a direct correction (discussed in the previous 
section) or fit a quadratic to the chemical potential edge. This latter 
method is advantageous in that it reflects that the contribution of 
miscalibration from anticipated sources are slowly varying in the detector
angle. It can also be used to give reasonably good behavior at the spectrometer
edges, where there might not be as much data as in the central region.

To build and apply a quadratic chemical potential correction in PyARPES,
you can use `arpes.corrections.build_quadratic_fermi_edge_correction`
and `arpes.corrections.apply_quadratic_fermi_edge_correction`. Here we 
will give an example of correcting the edge of a metal reference. 


![](static/quadratic-correction.png)

### Higher Dimensional Datasets

Curve fitting and shifting in PyARPES can be automatically broadcasted 
across extra dimensions. You should not need to worry about the 
dimensionality of your data. 

## Manual Corrections

If for some reason the built in corrections don't suit you, it is 
as easy to build and apply the correction yourself with 
`broadcast_model` and `.T.shift_by`. Here's an example

![](static/manual-correction.png)
  