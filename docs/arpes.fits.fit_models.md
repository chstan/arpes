# arpes.fits.fit\_models module

**class arpes.fits.fit\_models.XModelMixin(func, independent\_vars=None,
param\_names=None, missing='none', prefix='', name=None,**kws)\*\*

> Bases: `lmfit.model.Model`
> 
> **guess\_fit(data, params=None, weights=None,**kwargs)\*\*
> 
> > Params allows you to pass in hints as to what the values and bounds
> > on parameters should be. Look at the lmfit docs to get hints about
> > structure :param data: :param params: :param kwargs: :return:
> 
> **xguess(data,**kwargs)\*\*

**class
arpes.fits.fit\_models.FermiLorentzianModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**class arpes.fits.fit\_models.GStepBModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for fitting Fermi functions with a linear background
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**class arpes.fits.fit\_models.QuadraticModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for fitting a quadratic function
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**class
arpes.fits.fit\_models.ExponentialDecayCModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for fitting an exponential decay with a constant background
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**class
arpes.fits.fit\_models.LorentzianModel(independent\_vars=\['x'\],
prefix='', missing=None, name=None,**kwargs)\*\*

> Bases: , `lmfit.models.LorentzianModel`

**class arpes.fits.fit\_models.GaussianModel(independent\_vars=\['x'\],
prefix='', missing=None, name=None,**kwargs)\*\*

> Bases: , `lmfit.models.GaussianModel`

**class arpes.fits.fit\_models.VoigtModel(independent\_vars=\['x'\],
prefix='', missing=None, name=None,**kwargs)\*\*

> Bases: , `lmfit.models.VoigtModel`

**class arpes.fits.fit\_models.ConstantModel(independent\_vars=\['x'\],
prefix='', missing=None,**kwargs)\*\*

> Bases: , `lmfit.models.ConstantModel`

**class arpes.fits.fit\_models.LinearModel(independent\_vars=\['x'\],
prefix='', missing=None, name=None,**kwargs)\*\*

> Bases: , `lmfit.models.LinearModel`
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> > This is not implemented for all models, but is available for many of
> > the built-in models.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters
> > 
> > Should be implemented for each model subclass to run
> > self.make\_params(), update starting values and return a Parameters
> > object.
> > 
> > NotImplementedError

**class
arpes.fits.fit\_models.GStepBStandardModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for fitting Fermi functions with a linear background
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**class
arpes.fits.fit\_models.AffineBackgroundModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for an affine background
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> > This is not implemented for all models, but is available for many of
> > the built-in models.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters
> > 
> > Should be implemented for each model subclass to run
> > self.make\_params(), update starting values and return a Parameters
> > object.
> > 
> > NotImplementedError

**class
arpes.fits.fit\_models.FermiDiracModel(independent\_vars=\['x'\],
prefix='', missing='drop', name=None,**kwargs)\*\*

> Bases:
> 
> A model for the Fermi Dirac function
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters

**class arpes.fits.fit\_models.BandEdgeBModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for fitting a Lorentzian and background multiplied into the
> fermi dirac distribution
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a model.
> > 
> > This is not implemented for all models, but is available for many of
> > the built-in models.
> > 
> >   - data : array\_like  
> >     Array of data to use to guess parameter values.
> > 
> >   - \>\>\*\*\<\<kws : optional  
> >     Additional keyword arguments, passed to model function.
> > 
> > params : Parameters
> > 
> > Should be implemented for each model subclass to run
> > self.make\_params(), update starting values and return a Parameters
> > object.
> > 
> > NotImplementedError

**arpes.fits.fit\_models.gaussian\_convolve(model\_instance)**

> Produces a model that consists of convolution with a Gaussian kernel
> :param model\_instance: :return:
