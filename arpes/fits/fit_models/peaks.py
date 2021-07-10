"""Includes multi-peak model definitions."""

import lmfit as lf

from .x_model_mixin import XModelMixin
from .functional_forms import gaussian, affine_bkg, lorentzian, twolorentzian
from lmfit.models import update_param_vals

__all__ = ["TwoGaussianModel", "TwoLorModel"]


class TwoGaussianModel(XModelMixin):
    """A model for two gaussian functions with a linear background."""

    @staticmethod
    def twogaussian(
        x, center=0, t_center=0, width=1, t_width=1, amp=1, t_amp=1, lin_bkg=0, const_bkg=0
    ):
        """Two gaussians and an affine background."""
        return (
            gaussian(x, center, width, amp)
            + gaussian(x, t_center, t_width, t_amp)
            + affine_bkg(x, lin_bkg, const_bkg)
        )

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Sets physical constraints for peak width and other parameters."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.twogaussian, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Very simple heuristics for peak location."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%st_width" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorModel(XModelMixin):
    """A model for two gaussian functions with a linear background."""

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Sets physical constraints for peak width and other parameters."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(twolorentzian, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_gamma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Very simple heuristics for peak location."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%sgamma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%st_gamma" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
