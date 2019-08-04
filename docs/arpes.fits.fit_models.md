# arpes.fits.fit\_models module

**class arpes.fits.fit\_models.XModelMixin(func, independent\_vars=None,
param\_names=None, nan\_policy='raise', missing=None, prefix='',
name=None,**kws)\*\*

> Bases: `lmfit.model.Model`
> 
> A mixin providing curve fitting for xarray.DataArray instances.
> 
> This amounts mostly to making *lmfit* coordinate aware, and providing
> a translation layer between xarray and raw np.ndarray instances.
> 
> Subclassing this mixin as well as an lmfit Model class should
> bootstrap an lmfit Model to one that works transparently on xarray
> data.
> 
> Alternatively, you can use this as a model base in order to build new
> models.
> 
> The core method here is *guess\_fit* which is a convenient utility
> that performs both a *lmfit.Model.guess*, if available, before
> populating parameters and performing a curve fit.
> 
> \_\_add\_\_ and \_\_mul\_\_ are also implemented, to ensure that the
> composite model remains an instance of a subclass of this mixin.
> 
> `dimension_order = None`
> 
> **guess\_fit(data, params=None, weights=None, guess=True, debug=False,
> prefix\_params=True, transpose=False,**kwargs)\*\*
> 
> > Params allows you to pass in hints as to what the values and bounds
> > on parameters should be. Look at the lmfit docs to get hints about
> > structure :param data: :param params: :param kwargs: :return:
> 
> `n_dims = 1`
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
prefix='', nan\_policy='raise',**kwargs)\*\*

> Bases: , `lmfit.models.LorentzianModel`

**class arpes.fits.fit\_models.GaussianModel(independent\_vars=\['x'\],
prefix='', nan\_policy='raise',**kwargs)\*\*

> Bases: , `lmfit.models.GaussianModel`

**class arpes.fits.fit\_models.VoigtModel(independent\_vars=\['x'\],
prefix='', nan\_policy='raise',**kwargs)\*\*

> Bases: , `lmfit.models.VoigtModel`

**class arpes.fits.fit\_models.ConstantModel(independent\_vars=\['x'\],
prefix='', nan\_policy='raise',**kwargs)\*\*

> Bases: , `lmfit.models.ConstantModel`

**class arpes.fits.fit\_models.LinearModel(independent\_vars=\['x'\],
prefix='', nan\_policy='raise',**kwargs)\*\*

> Bases: , `lmfit.models.LinearModel`
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a Model.
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
> > Guess starting values for the parameters of a Model.
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
arpes.fits.fit\_models.AffineBroadenedFD(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for fitting an affine density of states with resolution
> broadened Fermi-Dirac occupation
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
> > Guess starting values for the parameters of a Model.
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

**class
arpes.fits.fit\_models.TwoGaussianModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for two gaussian functions with a linear background
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

**class arpes.fits.fit\_models.TwoLorModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for two gaussian functions with a linear background
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
arpes.fits.fit\_models.TwoLorEdgeModel(independent\_vars=\['x'\],
prefix='', missing='raise', name=None,**kwargs)\*\*

> Bases:
> 
> A model for (two lorentzians with an affine background) multiplied by
> a gstepb
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
arpes.fits.fit\_models.SplitLorentzianModel(independent\_vars=\['x'\],
prefix='', nan\_policy='raise',**kwargs)\*\*

> Bases: , `lmfit.models.SplitLorentzianModel`
> 
> **guess(data, x=None,**kwargs)\*\*
> 
> > Guess starting values for the parameters of a Model.
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
