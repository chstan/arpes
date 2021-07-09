"""Some miscellaneous model definitions."""

from lmfit.models import update_param_vals
import lmfit as lf
import numpy as np

from .x_model_mixin import XModelMixin

__all__ = [
    "QuadraticModel",
    "FermiVelocityRenormalizationModel",
    "LogRenormalizationModel",
]


class QuadraticModel(XModelMixin):
    """A model for fitting a quadratic function."""

    @staticmethod
    def quadratic(x, a=1, b=0, c=0):
        """Quadratic polynomial."""
        return a * x ** 2 + b * x + c

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Just defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.quadratic, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for parameter guesses."""
        pars = self.make_params()

        pars["%sa" % self.prefix].set(value=0)
        pars["%sb" % self.prefix].set(value=0)
        pars["%sc" % self.prefix].set(value=data.mean())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiVelocityRenormalizationModel(XModelMixin):
    """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials."""

    @staticmethod
    def fermi_velocity_renormalization_mfl(x, n0, v0, alpha, eps):
        """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials.

        Args:
            x: value to evaluate fit at (carrier density)
            n0: Value of carrier density at cutoff energy for validity of Dirac fermions
            v0: Bare velocity
            alpha: Fine structure constant
            eps: Graphene Dielectric constant
        """
        #     y = v0 * (rs/np.pi)*(5/3 + np.log(rs))+(rs/4)*np.log(kc/np.abs(kF))
        fx = v0 * (1 + (alpha / (1 + eps)) * np.log(n0 / np.abs(x)))
        fx2 = v0 * (1 + (alpha / (1 + eps * np.abs(x))) * np.log(n0 / np.abs(x)))
        fx3 = v0 * (1 + (alpha / (1 + eps * x ** 2)) * np.log(n0 / np.abs(x)))
        # return v0 + v0*(alpha/(8*eps))*np.log(n0/x)
        return fx3

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Sets physically reasonable constraints on parameter values."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.fermi_velocity_renormalization_mfl, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("n0", min=0.0)
        self.set_param_hint("eps", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for parameter estimation."""
        pars = self.make_params()

        # pars['%sn0' % self.prefix].set(value=10)
        # pars['%seps' % self.prefix].set(value=8)
        # pars['%svF' % self.prefix].set(value=(data.max()-data.min())/(kC-kD))

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class LogRenormalizationModel(XModelMixin):
    """A model for Logarithmic Renormalization to Linear Dispersion in Dirac Materials."""

    @staticmethod
    def log_renormalization(x, kF=1.6, kD=1.6, kC=1.7, alpha=0.4, vF=1e6):
        """Logarithmic correction to linear dispersion near charge neutrality in Dirac materials.

        As examples, this can be used to study the low energy physics in high quality ARPES spectra of graphene
        or topological Dirac semimetals.

        Args:
            x: The coorindates for the fit
            k: value to evaluate fit at
            kF: Fermi wavevector
            kD: Dirac point
            alpha: Fine structure constant
            vF: Bare Band Fermi Velocity
            kC: Cutoff Momentum
        """
        dk = x - kF
        dkD = x - kD
        return -vF * np.abs(dkD) + (alpha / 4) * vF * dk * np.log(np.abs(kC / dkD))

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """The fine structure constant and velocity must be nonnegative, so we will constrain them here."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.log_renormalization, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("vF", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for actually making parameter estimates here."""
        pars = self.make_params()

        pars["%skC" % self.prefix].set(value=1.7)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
