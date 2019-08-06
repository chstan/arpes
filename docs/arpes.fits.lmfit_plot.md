# arpes.fits.lmfit\_plot module

Monkeypatch the lmfit plotting to avoid TeX errors, and to allow
plotting model results in 2D.

This is a very safe monkey patch as we defer to the original plotting
function in cases where it is appropriate, rather than reimplementing
this functionality.

**arpes.fits.lmfit\_plot.patched\_plot(self, \*args,**kwargs)\*\*

> PyARPES patch for *lmfit* summary plots. Scientists like to have LaTeX
> in their plots, but because underscores outside TeX environments crash
> matplotlib renders, we need to do some fiddling with titles and axis
> labels in order to prevent users having to switch TeX on and off all
> the time.
> 
> Additionally, this patch provides better support for multidimensional
> curve fitting. :param self: :param args: :param kwargs: :return:

**arpes.fits.lmfit\_plot.transform\_lmfit\_titles(l, is\_title=False)**
