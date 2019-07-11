# Statistics in PyARPES

Ensuring accurate statistics in ARPES experiments is complicated 
by a large number of experimental degrees of freedom, sometimes short 
sample lifetimes in UHV or under x-ray irradiation, and the use of 
MCP/phosphor/CCD detectors which are strongly nonlinear under standard
operating conditions.

In the case where ARPES is conducted with a time of flight or a delay line detector,
single electron events allow for accurate accounting of statistics in a given detector channel.
For these types of experiments, PyARPES provides basic statistics support in the form 
of bootstrapping utilities.

## Parametric Bootstrap

Statistical bootstrapping allows construction of confidence intervals and statistics on some types of
estimators under appropriate conditions by resampling of a limited set of data. If the dataset 
is large enough, we can resample from the observed samples directly, yielding the 
nonparametric bootstrap.

In the case where we have strong reason to apply a particular model to the data, such as in 
counting experiments where fluctuations in the number of counts in a channel should be Poisson,
we can construct a maximum likelihood model and repeatedly draw samples from this model.

In order to bootstrap, we want to take any function `f` which consumes datasets and produces 
an estimator, and turn it into a function that takes datasets resamples this dataset, and produces 
a sampled distribution of values of the estimator. `arpes.bootstrap.bootstrap` does exactly this. 
Suppose we have a time-of-flight spin resolved ARPES spectrum and want to get a distribution 
of values for the total number of electrons that arrived during the timing window 185ns to 195ns 
(corresponding to the surface state at the chemical potential in this case).

![](/static/example-spectrum.png)

Now we can sum the number of counts in the surface state and request 5000 samples redrawn from 
a product of Poisson distributions&mdash;one for each channel.  

![](/static/bootstrap-ss.png)

A variety of utilities for doing other types of bootstraps is available in `arpes.bootstrap`, 
including for bootstrapping the spin polarization, error bars on the number of counts in each channel,
and more.