"""Wraps standard lmfit models."""
import lmfit as lf
import numpy as np

from lmfit.models import guess_from_peak, update_param_vals

from .x_model_mixin import XModelMixin

__all__ = [
    "VoigtModel",
    "GaussianModel",
    "ConstantModel",
    "LorentzianModel",
    "SkewedVoigtModel",
    "SplitLorentzianModel",
    "LinearModel",
    "LogisticModel",
    "StepModel",
]


class VoigtModel(XModelMixin, lf.models.VoigtModel):
    """Wraps `lf.models.VoigtModel`."""

    pass


class GaussianModel(XModelMixin, lf.models.GaussianModel):
    """Wraps `lf.models.GaussianModel`."""

    pass


class ConstantModel(XModelMixin, lf.models.ConstantModel):
    """Wraps `lf.models.ConstantModel`."""

    pass


class LorentzianModel(XModelMixin, lf.models.LorentzianModel):
    """Wraps `lf.models.LorentzianModel`."""

    pass


class SkewedVoigtModel(XModelMixin, lf.models.SkewedVoigtModel):
    """Wraps `lf.models.SkewedVoigtModel`."""

    pass


class SkewedGaussianModel(XModelMixin, lf.models.SkewedGaussianModel):
    """Wraps `lf.models.SkewedGaussianModel`."""

    pass


class SplitLorentzianModel(XModelMixin, lf.models.SplitLorentzianModel):
    """Wraps `lf.models.SplitLorentzianModel`."""

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()
        pars = guess_from_peak(self, data, x, negative=False, ampscale=1.25)
        sigma = pars["%ssigma" % self.prefix]
        pars["%ssigma_r" % self.prefix].set(value=sigma.value, min=sigma.min, max=sigma.max)

        return update_param_vals(pars, self.prefix, **kwargs)


class LinearModel(XModelMixin, lf.models.LinearModel):
    """A linear regression model."""

    def guess(self, data, x=None, **kwargs):
        """Use np.polyfit to get good initial parameters."""
        sval, oval = 0.0, 0.0
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(intercept=oval, slope=sval)
        return update_param_vals(pars, self.prefix, **kwargs)


class LogisticModel(XModelMixin, lf.models.StepModel):
    """A logistic regression model."""

    def __init__(self, independent_vars=["x"], prefix="", missing="raise", name=None, **kwargs):
        """Set standard parameters and delegate to lmfit."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
                "form": "logistic",
            }
        )
        super().__init__(**kwargs)


class StepModel(XModelMixin, lf.models.StepModel):
    """Wraps `lf.models.StepModel`."""

    pass
