import lmfit as lf
import numpy as np
import xarray as xr
import operator
import warnings

from scipy import stats
from lmfit.models import update_param_vals
from scipy.special import erfc
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from lmfit.models import guess_from_peak

from arpes.constants import HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ

__all__ = ('XModelMixin', 'FermiLorentzianModel','GStepBModel', 'QuadraticModel',
           'ExponentialDecayCModel', 'LorentzianModel', 'GaussianModel', 'VoigtModel',
           'ConstantModel', 'LinearModel', 'GStepBStandardModel', 'AffineBackgroundModel',
           'AffineBroadenedFD',
           'FermiDiracModel', 'BandEdgeBModel',
           'gaussian_convolve', 'TwoGaussianModel', 'TwoLorModel', 'TwoLorEdgeModel','SplitLorentzianModel')


class XModelMixin(lf.Model):
    n_dims = 1
    dimension_order = None

    def guess_fit(self, data, params=None, weights=None, guess=True, debug=False, prefix_params=True, transpose=False, **kwargs):
        """
        Params allows you to pass in hints as to what the values and bounds on parameters
        should be. Look at the lmfit docs to get hints about structure
        :param data:
        :param params:
        :param kwargs:
        :return:
        """
        if transpose:
            assert(len(data.dims) == 1 and "You cannot transpose (invert) a multidimensional array (scalar field).")

        coord_values = {}
        if 'x' in kwargs:
            coord_values['x'] = kwargs.pop('x')

        real_data, flat_data = data, data

        new_dim_order = None
        if isinstance(data, xr.DataArray):
            real_data, flat_data = data.values, data.values
            assert(len(real_data.shape) == self.n_dims)

            if self.n_dims == 1:
                coord_values['x'] = data.coords[list(data.indexes)[0]].values
            else:
                def find_appropriate_dimension(dim_or_dim_list):
                    if isinstance(dim_or_dim_list, str):
                        assert(dim_or_dim_list in data.dims)
                        return dim_or_dim_list

                    else:
                        intersect = set(dim_or_dim_list).intersection(data.dims)
                        assert(len(intersect) == 1)
                        return list(intersect)[0]

                # resolve multidimensional parameters
                new_dim_order = [find_appropriate_dimension(dim_options) for dim_options in self.dimension_order]
                if list(new_dim_order) != list(data.dims):
                    warnings.warn('Transposing data for multidimensional fit.')
                    data = data.transpose(*new_dim_order)

                coord_values = {k: v.values for k, v in data.coords.items() if k in new_dim_order}
                real_data, flat_data = data.values, data.values.ravel()

        real_weights = weights
        if isinstance(weights, xr.DataArray):
            if self.n_dims == 1:
                real_weights = real_weights.values
            else:
                if new_dim_order is not None:
                    real_weights = weights.transpose(*new_dim_order).values.ravel()
                else:
                    real_weights = weights.values.ravel()

        if transpose:
            cached_coordinate = list(coord_values.values())[0]
            coord_values[list(coord_values.keys())[0]] = real_data
            real_data = cached_coordinate
            flat_data = real_data

        if guess:
            guessed_params = self.guess(real_data, **coord_values)
        else:
            guessed_params = self.make_params()

        if params is not None:
            for k, v in params.items():
                if isinstance(v, dict):
                    if prefix_params:
                        guessed_params[self.prefix + k].set(**v)
                    else:
                        guessed_params[k].set(**v)
            guessed_params.update({self.prefix + k: v for k, v in params.items() if isinstance(v, lf.model.Parameter)})

        result = None
        try:
            result = super().fit(flat_data, guessed_params, **coord_values, weights=real_weights, **kwargs)
            result.independent = coord_values
            result.independent_order = new_dim_order
        except Exception as e:
            print(e)
            if debug:
                import pdb
                pdb.post_mortem(e.__traceback__)
        finally:
            return result

    def xguess(self, data, **kwargs):
        x = kwargs.pop('x', None)

        real_data = data
        if isinstance(data, xr.DataArray):
            real_data = data.values
            assert (len(real_data.shape) == 1)
            x = data.coords[list(data.indexes)[0]].values

        return self.guess(real_data, x=x, **kwargs)

    def __add__(self, other):
        """+"""
        return XAdditiveCompositeModel(self, other, operator.add)

    def __mul__(self, other):
        """*"""
        return XMultiplicativeCompositeModel(self, other, operator.mul)


class XAdditiveCompositeModel(lf.CompositeModel, XModelMixin):
    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars

class XMultiplicativeCompositeModel(lf.CompositeModel, XModelMixin):
    """
    Currently this just copies +, might want to adjust things!
    """
    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars

class XConvolutionCompositeModel(lf.CompositeModel, XModelMixin):
    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        guessed = {}

        for c in self.components:
            if c.prefix == 'conv_':
                # don't guess on the convolution term
                continue

            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars

def np_convolve(original_data, convolution_kernel):
    return convolve(original_data, convolution_kernel, mode='same')

def _convolve(original_data, convolution_kernel):
    n_points = min(len(original_data), len(convolution_kernel))
    padding = np.ones(n_points)
    temp = np.concatenate((padding * original_data[0], original_data, padding * original_data[-1]))
    convolved = np.convolve(temp, convolution_kernel, mode='valid')
    n_offset = int((len(convolved) - n_points) / 2)
    return (convolved[n_offset:])[:n_points]

def gaussian_convolve(model_instance):
    """
    Produces a model that consists of convolution with a Gaussian kernel
    :param model_instance:
    :return:
    """
    return XConvolutionCompositeModel(
        model_instance, GaussianModel(prefix='conv_'), convolve)


def effective_mass_bkg(eV, kp, m_star=0,
                       k_center=0, eV_center=0,
                       gamma=1, amplitude=1,
                       amplitude_k=0,
                       const_bkg=0, k_bkg=0, eV_bkg=0):
    """
    Model implementation function for simultaneous 2D curve fitting of band effective mass.
    Allows for an affine background in each dimension, together with variance in the band intensity
    along the band, as a very simple model of matrix elements. Together with prenormalizing your data
    this should allow reasonable fits of a lot of typical ARPES data.
    :param eV:
    :param k:
    :param m_star:
    :param k_center:
    :param eV_center:
    :param gamma:
    :param amplitude:
    :param amplitude_k:
    :param const_bkg:
    :param k_bkg:
    :param eV_bkg:
    :return:
    """
    bkg = np.outer(eV * 0 + 1, k_bkg * kp) + np.outer(eV_bkg * eV, kp * 0 + 1) + const_bkg

    # check units
    dk = kp - k_center
    offset = HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ * dk ** 2 / (2 * m_star + 1e-6)
    eVk = np.outer(eV, kp * 0 + 1)
    coherent = (amplitude + amplitude_k * dk) * (1 / (2 * np.pi)) * gamma / (
    (eVk - eV_center + offset) ** 2 + (0.5 * gamma) ** 2)
    return (coherent + bkg).ravel()


def affine_bkg(x, lin_bkg=0, const_bkg=0):
    return lin_bkg * x + const_bkg


def quadratic(x, a=1, b=0, c=0):
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

def gstepb_standard(x, center=0, sigma=1, amplitude=1, **kwargs):
    return gstepb(x, center, width=sigma, erf_amp=amplitude, **kwargs)

def exponential_decay_c(x, amp, tau, t0, const_bkg):
    dx = x - t0
    mask = (dx >= 0) * 1
    return const_bkg + amp * mask * np.exp(-(x - t0)/tau)

def twoexponential_decay_c(x,amp,t0,tau1,tau2,const_bkg):
    dx = x-t0
    mask = (dx >= 0) * 1
    y = const_bkg + amp*(1-np.exp(-dx/tau1))*np.exp(-dx/tau2)
    f = y.copy()
    f[dx<0] = const_bkg
    f[dx>=0] = y[dx>=0]
    return f


def lorentzian(x, gamma, center, amplitude):
    return amplitude * (1/(2*np.pi))* gamma /((x-center)**2+(.5*gamma)**2)


def pseudo_shirley(x, gamma, center, amplitude, bkg_amplitude):
    lor = lorentzian(x, gamma, center, amplitude)


def twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg):
    L1 = lorentzian(x, gamma, center, amp)
    L2 = lorentzian(x, t_gamma, t_center, t_amp)
    AB = affine_bkg(x, lin_bkg, const_bkg)
    return L1 + L2 + AB


def gstepb_mult_lorentzian(x, center=0, width=1, erf_amp=1, lin_bkg=0, const_bkg=0, gamma=1, lorcenter=0):
    return gstepb(x, center, width, erf_amp, lin_bkg, const_bkg)*lorentzian(x, gamma, lorcenter, 1)


def fermi_dirac(x, center=0, width=0.05, scale=1):
    # Fermi edge
    return scale / (np.exp((x - center) / width) + 1)


def fermi_dirac_bkg(x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1):
    # Fermi edge with an affine background multiplied in
    return (scale + lin_bkg) / (np.exp((x - center) / width) + 1) + const_bkg


def band_edge_bkg(x, center=0, width=0.05, amplitude=1, gamma=0.1, lor_center=0, offset=0, lin_bkg=0, const_bkg=0):
    # Lorentzian plus affine background multiplied into fermi edge with overall offset
    return (lorentzian(x, gamma, lor_center, amplitude) + lin_bkg * x + const_bkg) * fermi_dirac(x, center, width) + offset


def lorentzian_affine(x, gamma=1, lor_center=0, amplitude=1, lin_bkg=0, const_bkg=0):
    return (lorentzian(x, gamma, lor_center, amplitude) + lin_bkg * x + const_bkg) 


def gaussian(x, center=0, sigma=1, amplitude=1):
    return amplitude*np.exp(-(x-center)**2/(2*sigma**2))


def twogaussian(x, center=0, t_center=0, width=1, t_width=1, amp=1, t_amp=1, lin_bkg=0, const_bkg=0):
    return gaussian(x, center, width, amp) + gaussian(x, t_center, t_width, t_amp) + affine_bkg(x, lin_bkg, const_bkg)


def twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg):
    L1 = lorentzian(x, gamma, center, amp)
    L2 = lorentzian(x, t_gamma, t_center, t_amp)
    AB = affine_bkg(x, lin_bkg, const_bkg)
    return L1 + L2 + AB

def twolorentzian_gstep(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg, g_center, sigma, erf_amp):
    TL = twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg)
    GS = gstep(x, g_center, sigma, erf_amp)
    return TL*GS

def affine_broadened_fd(x, fd_center=0, fd_width=0.003, conv_width=0.02, const_bkg=1, lin_bkg=0, offset=0):
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
    dx = x - fd_center
    x_scaling = x[1] - x[0]
    fermi = 1 / (np.exp(dx / fd_width) + 1)
    return gaussian_filter(
        (const_bkg + lin_bkg * dx) * fermi,
        sigma=conv_width / x_scaling
    ) + offset


# Nick Dale edits

def log_renormalization(x, kF=1.6,kD=1.6,kC= 1.7, alpha = 0.4, vF=1e6):
    """
    Logarithmic Correction to Linear Dispersion of Materials with Dirac Dispersion near charge neutrality
    :param k: value to evaluate fit at
    :param kF: Fermi wavevector
    :param kD: Dirac point
    :param alpha: Fine structure constant
    :param vF: Bare Band Fermi Velocity
    :param kC: Cutoff Momentum
    """
    dk = x - kF
    dkD = x - kD
    return -vF*np.abs(dkD)+(alpha/4)*vF*dk*np.log(np.abs(kC/dkD))

def dirac_dispersion(x, kd = 1.6, amplitude_1 = 1,amplitude_2=1,center=0, sigma_1 = 1,sigma_2 =1):
    """
    Model for dirac_dispersion symmetric about the dirac point. Fits lorentziants to (kd-center) and (kd+center)
    :param x: value to evaluate fit at
    :param kd: Dirac point momentum
    :param amplitude_1: amplitude of Lorentzian at kd-center
    :param amplitude_2: amplitude of Lorentzian at kd+center
    :param center: center of Lorentzian
    :param sigma_1: FWHM of Lorentzian at kd-center
    :param sigma_2: FWHM of Lorentzian at kd+center

    """

    dx = x-center
    return lorentzian(x,center=kd-center,amplitude=amplitude_1,gamma=sigma_1) + lorentzian(x,center=kd+center,amplitude=amplitude_2,gamma=sigma_2)

# end Nick Dale edits




# Daniel Eilbott edits
def g(x, mu=0, sigma=0.1):
    return (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(1/2) * ((x-mu) / sigma)**2)

def band_edge_bkg_gauss(x, center=0, width=0.05, amplitude=1, gamma=0.1, lor_center=0, offset=0, lin_bkg=0, const_bkg=0):#,sigma=0.1):
    return np_convolve(band_edge_bkg(x, 0, width, amplitude, gamma, lor_center, offset, lin_bkg, const_bkg), g(np.linspace(-6,6,800), 0, 0.01))

def fermi_dirac_affine(x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1):
    # Fermi edge with an affine background multiplied in
    return (scale + lin_bkg * x) / (np.exp((x - center) / width) + 1) + const_bkg

def fermi_dirac_bkg_gauss(x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1, sigma=0.01):
    return np_convolve(
        fermi_dirac_affine(x, center, width, lin_bkg, const_bkg, scale),
        # g(np.linspace(-6,6,800),0,sigma)
        g(x,(min(x)+max(x))/2,sigma)
    )

def gstep_stdev(x, center=0, sigma=1, erf_amp=1):
    """
    Fermi function convolved with a Gaussian
    :param x: value to evaluate fit at
    :param center: center of the step
    :param width: width of the step
    :param erf_amp: height of the step
    :return:
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(np.sqrt(2) * dx / width)

def gstepb_stdev(x, center=0, sigma=1, erf_amp=1, lin_bkg=0, const_bkg=0):
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
    return const_bkg + lin_bkg * np.min(dx, 0) + gstep_stdev(x, center, sigma, erf_amp)

# / Daniel Eilbott


class EffectiveMassModel(XModelMixin):
    """
    A two dimensional model for a quadratic distribution of Lorentzians
    """
    n_dims = 2
    dimension_order = ['eV', ['kp', 'phi']]

    def __init__(self, independent_vars=['eV', 'kp'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(effective_mass_bkg, **kwargs)

        self.set_param_hint('gamma', min=0.)
        self.set_param_hint('amplitude', min=0.)

    def guess(self, data, eV=None, kp=None, phi=None, **kwargs):
        momentum = kp if kp is not None else phi
        try:
            momentum = momentum.values
            eV = eV.values
            data = data.values
        except AttributeError:
            pass

        pars = self.make_params()

        pars['%sm_star' % self.prefix].set(value=1)

        pars['%sk_center' % self.prefix].set(value=np.mean(momentum))
        pars['%seV_center' % self.prefix].set(value=np.mean(eV))

        pars['%samplitude' % self.prefix].set(value=np.mean(np.mean(data, axis=0)))
        pars['%sgamma' % self.prefix].set(value=0.25)

        pars['%samplitude_k' % self.prefix].set(value=0)  # can definitely improve here

        # Crude estimate of the background
        left, right = np.mean(data[:5,:], axis=0), np.mean(data[-5:,:], axis=0)
        top, bottom = np.mean(data[:,:5], axis=0), np.mean(data[:,-5:], axis=0)
        left, right = np.percentile(left, 10), np.percentile(right, 10)
        top, bottom = np.percentile(top, 10), np.percentile(bottom, 10)

        pars['%sconst_bkg' % self.prefix].set(value=np.min(np.array([left, right, top, bottom])))
        pars['%sk_bkg' % self.prefix].set(value=(bottom - top) / (eV[-1] - eV[0]))
        pars['%seV_bkg' % self.prefix].set(value=(right - left) / (momentum[-1] - momentum[0]))

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class AffineBroadenedFD(XModelMixin):
    """
    A model for fitting an affine density of states with resolution broadened Fermi-Dirac occupation
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(affine_broadened_fd, **kwargs)

        self.set_param_hint('offset', min=0.)
        self.set_param_hint('fd_width', min=0.)
        self.set_param_hint('conv_width', min=0.)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%sfd_center' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.mean().item() * 2)
        pars['%soffset' % self.prefix].set(value=data.min().item())

        pars['%sfd_width' % self.prefix].set(0.005)  # TODO we can do better than this
        pars['%sconv_width' % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiLorentzianModel(XModelMixin):
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(gstepb_mult_lorentzian, **kwargs)

        self.set_param_hint('erf_amp', min=0.)
        self.set_param_hint('width', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)
        self.set_param_hint('gamma', min=0.)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%slorcenter' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%swidth' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%serf_amp' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiDiracModel(XModelMixin):
    """
    A model for the Fermi Dirac function
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='drop', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(fermi_dirac, **kwargs)

        self.set_param_hint('width', min=0)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['{}center'.format(self.prefix)].set(value=0)
        pars['{}width'.format(self.prefix)].set(value=0.05)
        pars['{}scale'.format(self.prefix)].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC

class GStepBModel(XModelMixin):
    """
    A model for fitting Fermi functions with a linear background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(gstepb, **kwargs)

        self.set_param_hint('erf_amp', min=0.)
        self.set_param_hint('width', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%swidth' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%serf_amp' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoBandEdgeBModel(XModelMixin):
    """
        A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution
        """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({
            'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars,
        })
        super().__init__(two_band_edge_bkg, **kwargs)

        self.set_param_hint('amplitude_1', min=0.)
        self.set_param_hint('gamma_1', min=0.)
        self.set_param_hint('amplitude_2', min=0.)
        self.set_param_hint('gamma_2', min=0.)

        self.set_param_hint('offset', min=-10)

    def guess(self, data, x=None, **kwargs):
        # should really do some peak fitting or edge detection to find
        # okay values here
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars['%slor_center' % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars['%slor_center' % self.prefix].set(value=-0.2)

        pars['%sgamma' % self.prefix].set(value=0.2)
        pars['%samplitude' % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%soffset' % self.prefix].set(value=data.min())

        pars['%scenter' % self.prefix].set(value=0)
        pars['%swidth' % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBModel(XModelMixin):
    """
        A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution
        """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({
            'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars,
        })
        super().__init__(band_edge_bkg, **kwargs)

        self.set_param_hint('amplitude', min=0.)
        self.set_param_hint('gamma', min=0.)
        self.set_param_hint('offset', min=-10)

    def guess(self, data, x=None, **kwargs):
        # should really do some peak fitting or edge detection to find
        # okay values here
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars['%slor_center' % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars['%slor_center' % self.prefix].set(value=-0.2)

        pars['%sgamma' % self.prefix].set(value=0.2)
        pars['%samplitude' % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%soffset' % self.prefix].set(value=data.min())

        pars['%scenter' % self.prefix].set(value=0)
        pars['%swidth' % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBGModel(XModelMixin):
    """
        A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution
        """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({
            'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars,
        })
        super().__init__(band_edge_bkg_gauss, **kwargs)

        self.set_param_hint('amplitude', min=0.)
        self.set_param_hint('gamma', min=0.)
        self.set_param_hint('offset', min=-10)
        self.set_param_hint('center', vary=False)

    def guess(self, data, x=None, **kwargs):
        # should really do some peak fitting or edge detection to find
        # okay values here
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars['%slor_center' % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars['%slor_center' % self.prefix].set(value=-0.2)

        pars['%sgamma' % self.prefix].set(value=0.2)
        pars['%samplitude' % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%soffset' % self.prefix].set(value=data.min())

        # pars['%scenter' % self.prefix].set(value=0)
        pars['%swidth' % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class FermiDiracAffGaussModel(XModelMixin):
    """
    A model for the Fermi Dirac function with an affine background multiplied, then all convolved with a Gaussian
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='drop', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(fermi_dirac_bkg_gauss, **kwargs)

        # self.set_param_hint('width', min=0)
        self.set_param_hint('width', vary=False)
        # self.set_param_hint('lin_bkg', max=10)
        # self.set_param_hint('scale', max=50000)
        self.set_param_hint('scale', min=0)
        self.set_param_hint('sigma', min=0, vary=True)
        self.set_param_hint('lin_bkg', vary=False)
        self.set_param_hint('const_bkg', vary=False)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['{}center'.format(self.prefix)].set(value=0)
        # pars['{}width'.format(self.prefix)].set(value=0.05)
        pars['{}width'.format(self.prefix)].set(value=0.0009264)
        pars['{}scale'.format(self.prefix)].set(value=data.mean() - data.min())
        pars['{}lin_bkg'.format(self.prefix)].set(value=0)
        pars['{}const_bkg'.format(self.prefix)].set(value=0)
        pars['{}sigma'.format(self.prefix)].set(value=0.023)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStdevModel(XModelMixin):
    """
    A model for fitting Fermi functions with a linear background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(gstepb_stdev, **kwargs)

        self.set_param_hint('erf_amp', min=0.)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%ssigma' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%serf_amp' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStandardModel(XModelMixin):
    """
    A model for fitting Fermi functions with a linear background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(gstepb_standard, **kwargs)

        self.set_param_hint('amplitude', min=0.)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%ssigma' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%samplitude' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class ExponentialDecayCModel(XModelMixin):
    """
    A model for fitting an exponential decay with a constant background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(exponential_decay_c, **kwargs)

        # amp is also a parameter, but we have no hint for it
        self.set_param_hint('tau', min=0.)
        # t0 is also a parameter, but we have no hint for it
        self.set_param_hint('const_bkg')

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%stau' % self.prefix].set(value=0.2) # 200fs
        pars['%st0' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.mean())
        pars['%samp' % self.prefix].set(value=data.max() - data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoExponentialDecayCModel(XModelMixin):
    """
    A model for fitting an exponential decay with a constant background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(twoexponential_decay_c, **kwargs)

        # amp is also a parameter, but we have no hint for it
        self.set_param_hint('tau1', min=0.)
        self.set_param_hint('tau2', min=0.)
        # t0 is also a parameter, but we have no hint for it
        self.set_param_hint('const_bkg')

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%stau1' % self.prefix].set(value=0.2) # 200fs
        pars['%stau2' % self.prefix].set(value=1) # 1ps
        pars['%st0' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.mean())
        pars['%samp' % self.prefix].set(value=data.max() - data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC    


class QuadraticModel(XModelMixin):
    """
    A model for fitting a quadratic function
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(quadratic, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%sa' % self.prefix].set(value=0)
        pars['%sb' % self.prefix].set(value=0)
        pars['%sc' % self.prefix].set(value=data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class AffineBackgroundModel(XModelMixin):
    """
    A model for an affine background
    """

    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(affine_bkg, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%slin_bkg' % self.prefix].set(value=np.percentile(data, 10))
        pars['%sconst_bkg' % self.prefix].set(value=0)

        return update_param_vals(pars, self.prefix, **kwargs)


class TwoGaussianModel(XModelMixin):
    """
    A model for two gaussian functions with a linear background
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(twogaussian, **kwargs)

        self.set_param_hint('amp', min=0.)
        self.set_param_hint('width', min=0)
        self.set_param_hint('t_amp', min=0.)
        self.set_param_hint('t_width', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%st_center' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%swidth' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%st_width' % self.prefix].set(0.02)
        pars['%samp' % self.prefix].set(value=data.mean() - data.min())
        pars['%st_amp' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorModel(XModelMixin):
    """
    A model for two gaussian functions with a linear background
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(twolorentzian, **kwargs)

        self.set_param_hint('amp', min=0.)
        self.set_param_hint('gamma', min=0)
        self.set_param_hint('t_amp', min=0.)
        self.set_param_hint('t_gamma', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%st_center' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%sgamma' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%st_gamma' % self.prefix].set(0.02)
        pars['%samp' % self.prefix].set(value=data.mean() - data.min())
        pars['%st_amp' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorEdgeModel(XModelMixin):
    """
    A model for (two lorentzians with an affine background) multiplied by a gstepb
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(twolorentzian_gstep, **kwargs)

        self.set_param_hint('amp', min=0.)
        self.set_param_hint('gamma', min=0)
        self.set_param_hint('t_amp', min=0.)
        self.set_param_hint('t_gamma', min=0)
        self.set_param_hint('erf_amp', min=0.)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('lin_bkg', min=-10, max=10)
        self.set_param_hint('const_bkg', min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%scenter' % self.prefix].set(value=0)
        pars['%st_center' % self.prefix].set(value=0)
        pars['%sg_center' % self.prefix].set(value=0)
        pars['%slin_bkg' % self.prefix].set(value=0)
        pars['%sconst_bkg' % self.prefix].set(value=data.min())
        pars['%sgamma' % self.prefix].set(0.02)  # TODO we can do better than this
        pars['%st_gamma' % self.prefix].set(0.02)
        pars['%ssigma' % self.prefix].set(0.02)
        pars['%samp' % self.prefix].set(value=data.mean() - data.min())
        pars['%st_amp' % self.prefix].set(value=data.mean() - data.min())
        pars['%serf_amp' % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC

class Log_Renormalization_Model(XModelMixin):
    """
    A model for Logarithmic Renormalization to Linear Dispersion in Dirac Materials
    :param k: value to evaluate fit at
    :param kF: Fermi wavevector
    :param kD: Dirac point
    :param alpha: Fine structure constant
    :param vF: Bare Band Fermi Velocity
    :param kC: Cutoff Momentum
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(log_renormalization, **kwargs)

        self.set_param_hint('alpha', min=0.)
        self.set_param_hint('vF', min=0.)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars['%skC' % self.prefix].set(value=1.7)
        # pars['%svF' % self.prefix].set(value=(data.max()-data.min())/(kC-kD))

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC

class DiracDispersionModel(XModelMixin):
    """
    Model for dirac_dispersion symmetric about the dirac point. Fits lorentziants to (kd-center) and (kd+center)
    :param x: value to evaluate fit at
    :param kd: Dirac point momentum
    :param amplitude_1: amplitude of Lorentzian at kd-center
    :param amplitude_2: amplitude of Lorentzian at kd+center
    :param center: center of Lorentzian
    :param sigma_1: FWHM of Lorentzian at kd-center
    :param sigma_2: FWHM of Lorentzian at kd+center
    """
    def __init__(self, independent_vars=['x'], prefix='', missing='raise', name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'independent_vars': independent_vars})
        super().__init__(dirac_dispersion, **kwargs)

        self.set_param_hint('sigma_1', min=0.)
        self.set_param_hint('sigma_2', min=0.)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        # pars['%skd' % self.prefix].set(value=1.5)
        

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC

class LorentzianModel(XModelMixin, lf.models.LorentzianModel):
    pass

class SplitLorentzianModel(XModelMixin, lf.models.SplitLorentzianModel):
    def _set_paramhints_prefix(self):
        """
        Conrad: In lmfit v0.9.13 there is a bug here where the prefix is not set 
        on the parameter hint expressions for the computed variables "height" 
        and "fwhm". Until this is patched, we need to do this correctly, which invovles
        just injecting the prefix in before each of the model parameters in these
        expressions.
        """
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('sigma_r', min=0)
        self.set_param_hint('fwhm', expr='{prefix}sigma+{prefix}sigma_r'.format(
            prefix=self.prefix))
        self.set_param_hint(
            'height', expr='2*{prefix}amplitude/{:.7f}/max({}, ({prefix}sigma+{prefix}sigma_r))'.format(
                np.pi, np.finfo(np.float).eps, prefix=self.prefix))

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params() 
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative=False, ampscale=1.25)
        sigma = pars['%ssigma' % self.prefix]
        pars['%ssigma_r' % self.prefix].set(value=sigma.value, min=sigma.min, max=sigma.max)

        return update_param_vals(pars, self.prefix, **kwargs)

class VoigtModel(XModelMixin, lf.models.VoigtModel):
    pass


class GaussianModel(XModelMixin, lf.models.GaussianModel):
    pass



class ConstantModel(XModelMixin, lf.models.ConstantModel):
    pass


class LinearModel(XModelMixin, lf.models.LinearModel):
    def guess(self, data, x=None, **kwargs):
        sval, oval = 0., 0.
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(intercept=oval, slope=sval)
        return update_param_vals(pars, self.prefix, **kwargs)
