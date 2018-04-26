## Science/Understanding Related

1. Ask about the significance of pass energy
2. Ask about undulator gap
3. Ask about noise suppression and how to handle windowing on the analyzer

## Data Formatting

1. Investigate differences between recorded energy in FITS under key SS_HV
2. Figure out lens mode names under key SSLNM0
3. Get a reference for the number of pixels per degree

## Optimizations

1. Profile the code in 
2. See if Numba would be appropriate in any places

## Needed for analysis

2. Implement a Python VSNR package according to [VSNR paper](https://www.math.univ-toulouse.fr/~weiss/Publis/IEEEIP_VSNR_Final.pdf)
3. Experiment with different denoising methods from scikit-image see 
[sk-image Tomography](http://emmanuelle.github.io/segmentation-of-3-d-tomography-images-with-python-and-scikit-image.html)
4. Build Lucy-Richardson deconvolution into analysis pipelines optionally
5. Make sure conversion functions can still work when we take a slice in constant energy. 
This will require looking at the energy attribute on the array if it is not passed
6. Change the pipeline scripts to turn functions into generators so that StopIteration can be produced in order to signify
that the pipeline is finished or cannot otherwise proceed. Or consider another method of achieving this
7. Be able to plot another scans' coordinates over a FS or similar
8. Total variation convex denoising to remove multiplicative noise from grid
9. Figure out coordinate projecting so that we can use the symmetry points
in reference maps in order to get offsets for subsequent scans 
10. Fix hv reference plots
11. Make sure conversion function also forward convert symmetry points, other scalar parameters
12. Data rotations
13. Waypoint path selections
16. Ability to cache named computations
17. Add verification for install 


## Needed for plotting

1. Plot MDC with stats over kspace map
2. Plot EDC with stats over kspace map
4. Surf plots

## Tests

1. Get something set up to test full analysis process against a test dataset
2. Performance tests/benchmarks
3. Tests should enforce API guarantees if more people start using this
4. Finish writing test fixtures for setting up configuration

## Today

1. Forward conversion
4. Fix curvature code
8. Continue work on installation integrity checker
9. Order ZnSe window, order feedthrough, order integrating photodetector
10. Order inlet screen
11. Do BL data analysis -- looks crappy!
14. Self energy calculation
15. Inner potential utility
16. Automatic inference of reference from dataset
17. Better Brillouin zone displays
18. Lucy Richardson testing
19. Fit broadcasting across extra axes


## Random Ideas

Incommensurate spin spirals observable in ARPES because of the finite probe depth? 
Same idea as in BSSCO

## Hardware

1. Knife puck for UV beam profiling, or gold edge patterned onto an insulator
2. Knife edge on manipulator for main chamber
3. Brush contacts gating main chamber
4. Replace ZnSe window on next bake main chamber
5. Replace ZnSe viewport + UV viewport with Ts on next bake main chamber
6. Better story about sample imaging and orientation in situ, look into lenses and mounts from Navitar + Thorlabs
7. Can get better beam focusing using a more complicated lens setup (blow up the beam first to 15mm, can get to 10 micron)