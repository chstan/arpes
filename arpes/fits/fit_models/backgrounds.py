"""Definitions of common backgrounds."""

from lmfit.models import update_param_vals
import numpy as np

from .x_model_mixin import XModelMixin
from .functional_forms import affine_bkg

__all__ = ["AffineBackgroundModel"]


class AffineBackgroundModel(XModelMixin):
    """A model for an affine background."""

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(affine_bkg, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Use the tenth percentile value for the slope and a zero offset.

        Generally this should converge well regardless.
        """
        pars = self.make_params()

        pars["%slin_bkg" % self.prefix].set(value=np.percentile(data, 10))
        pars["%sconst_bkg" % self.prefix].set(value=0)

        return update_param_vals(pars, self.prefix, **kwargs)
