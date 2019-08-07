import numpy as np

from arpes.fits import LinearModel
from arpes.typing import DataType

__all__ = ('kfermi_from_mdcs',)


def kfermi_from_mdcs(mdc_results: DataType, param=None):
    """
    Calculates a Fermi momentum using a series of MDCs and the known
    Fermi level (eV=0). This is especially useful to
    isolate an area for analysis.

    This method tolerates data that came from a prefixed model fit,
    but will always attempt to look for an attribute containing "center".

    :param mdc_results: A DataArray or Dataset containing `lmfit.ModelResult`s.
    :param param:
    :return:
    """
    real_param_name = None

    param_names = mdc_results.results.F.parameter_names

    if param in param_names and param is not None:
        real_param_name = param
    else:
        best_names = [p for p in param_names if 'center' in p]
        if param is not None:
            best_names = [p for p in best_names if param in p]

        assert len(best_names) == 1
        real_param_name = best_names[0]

    def nan_sieve(_, x):
        return not np.isnan(x.item())

    return LinearModel().guess_fit(
        mdc_results.F.p(real_param_name).T.filter_coord('eV', nan_sieve)).eval(x=0)
