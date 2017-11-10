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

## Needed for plotting

1. Plot MDC with stats over kspace map
2. Plot EDC with stats over kspace map

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

