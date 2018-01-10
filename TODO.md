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

1. Bootstrap resampling for statistical errors
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

## Needed for plotting

1. Plot MDC with stats over kspace map
2. Plot EDC with stats over kspace map
3. Bootstrap resampling (see also analysis)

## Stretch Goals/Tidyness

1. Warnings for data discrepancy between dataset JSON and FITS headers

## Documentation

1. Add module level docstrings
2. Add class level docstrings
3. Look into a documentation generator like Sphinx

## Tests

1. Get something set up to test full analysis process against a test dataset
2. Performance tests/benchmarks
3. Tests should enforce API guarantees if more people start using this
