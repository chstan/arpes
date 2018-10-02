from arpes.typing import DataType
import numpy as np

from arpes.fits import LinearModel

__all__ = ('kfermi_from_mdcs',)


def kfermi_from_mdcs(mdc_results: DataType, param=None):
    real_param_name = None

    param_names = mdc_results.results.F.parameter_names

    if param in param_names and param is not None:
        real_param_name = param
    else:
        best_names = [p for p in param_names if 'center' in p]
        if param is not None:
            best_names = [p for p in best_names if param in p]

        assert(len(best_names) == 1)
        real_param_name = best_names[0]

    def nan_sieve(_, x):
        return not np.isnan(x.item())

    return LinearModel().guess_fit(
        mdc_results.F.p(real_param_name).T.filter_coord('eV', nan_sieve)).eval(x=0)
