"""Definitions of models involving Dirac points, graphene, graphite."""

from lmfit.models import update_param_vals
import lmfit as lf

from .x_model_mixin import XModelMixin
from .functional_forms import lorentzian

__all__ = [
    "DiracDispersionModel",
]


class DiracDispersionModel(XModelMixin):
    """Model for dirac_dispersion symmetric about the dirac point."""

    def dirac_dispersion(x, kd=1.6, amplitude_1=1, amplitude_2=1, center=0, sigma_1=1, sigma_2=1):
        """Model for dirac_dispersion symmetric about the dirac point.

        Fits lorentziants to (kd-center) and (kd+center)

        Args:
            x: value to evaluate fit at
            kd: Dirac point momentum
            amplitude_1: amplitude of Lorentzian at kd-center
            amplitude_2: amplitude of Lorentzian at kd+center
            center: center of Lorentzian
            sigma_1: FWHM of Lorentzian at kd-center
            sigma_2: FWHM of Lorentzian at kd+center

        Returns:
            An MDC model for a Dirac like dispersion around the cone.
        """
        return lorentzian(x, center=kd - center, amplitude=amplitude_1, gamma=sigma_1) + lorentzian(
            x, center=kd + center, amplitude=amplitude_2, gamma=sigma_2
        )

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.dirac_dispersion, **kwargs)

        self.set_param_hint("sigma_1", min=0.0)
        self.set_param_hint("sigma_2", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        # pars['%skd' % self.prefix].set(value=1.5)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
