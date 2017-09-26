import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import update_param_vals
from scipy.special import erfc

def quadratic(x, a = 1, b = 0, c = 0):
    return a * x**2 + b * x + c

def gstepb(x, center=0, width=1, erf_amp=1, lin_bkg=0, const_bkg=0):
    """
    Fermi function convoled with a Gaussian together with affine background
    :param x: value to evaluate function at
    :param center: center of the step
    :param width: width of the step
    :param erf_amp: height of the step
    :param lin_bkg: linear background slope
    :param const_bkg: constant background
    :return:
    """
    dx = x - center
    return const_bkg + lin_bkg * np.min(dx, 0) + gstep(x, center, width, erf_amp)

def gstep(x, center=0, width=1, erf_amp=1):
    """
    Fermi function convolved with a Gaussian
    :param x: value to evaluate fit at
    :param center: center of the step
    :param width: width of the step
    :param erf_amp: height of the step
    :return:
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(1.66511 * dx / width)


class XModel(lf.Model):
    def __init__(self, func, **kwargs):
        super(XModel, self).__init__(func, **kwargs)

    def guess_fit(self, data, params=None, **kwargs):
        x = kwargs.pop('x', None)

        real_data = data
        if isinstance(data, xr.DataArray):
            real_data = data.values
            assert(len(real_data.shape) == 1)
            x = data.coords[list(data.coords)[0]].values

        guessed_params = self.guess(real_data)
        if params is not None:
            guessed_params.update(params)

        return super(XModel, self).fit(real_data, guessed_params, x=x, **kwargs)

class GStepBModel(XModel):
    """
    A model for fitting Fermi functions with a linear background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super(GStepBModel, self).__init__(gstepb, **kwargs)

        self.set_param_hint('erf_amp', min=0.)
        self.set_param_hint('width', min=0)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=0)
        pars['%swidth' % self.prefix].set(0.03) # TODO we can do better than this
        pars['%serf_amp' % self.prefix].set(value=data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC

class QuadraticModel(XModel):
    """
    A model for fitting a quadratic function
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super(QuadraticModel, self).__init__(quadratic, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%sa' % self.prefix].set(value=0)
        pars['%sb' % self.prefix].set(value=0)
        pars['%sc' % self.prefix].set(value=data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC

def broadcast_model(model_cls: type, dataset: xr.DataArray, broadcast_axis):
    cs = {}
    cs[broadcast_axis] = dataset.coords[broadcast_axis]

    model = model_cls()
    broadcast_values = dataset.coords[broadcast_axis].values
    fit_results = [model.guess_fit(dataset.sel(**dict([(broadcast_axis, v,)])))
                   for v in broadcast_values]

    return xr.DataArray(
        np.asarray(fit_results),
        coords=cs,
        dims=[broadcast_axis],
    )