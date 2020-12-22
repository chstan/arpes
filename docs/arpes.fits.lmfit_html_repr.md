arpes.fits.lmfit\_html\_repr module
===================================

For now we monkeypatch lmfit to make it easier to work with in Jupyter.
We should consider forking or providing a pull at a later date after
this settles down.

The end goal here is to allow pleasing and functional representations of
curve fitting sessions performed in Jupyter, so that they can be rapidly
understood, and screencapped for simple purposes, like including in
group meeting notes.

**arpes.fits.lmfit\_html\_repr.repr\_html\_Model(self)**

> Better Jupyter representation of *lmfit.Model* instances. :param self:
> :return:

**arpes.fits.lmfit\_html\_repr.repr\_html\_ModelResult(self,**kwargs)\*\*

> Provides a better Jupyter representation of an *lmfit.ModelResult*
> instance. :param self: :param kwargs: :return:

**arpes.fits.lmfit\_html\_repr.repr\_html\_Parameter(self,
short=False)**

**arpes.fits.lmfit\_html\_repr.repr\_html\_Parameters(self,
short=False)**
