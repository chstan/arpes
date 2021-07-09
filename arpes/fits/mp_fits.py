"""Worker for process parallel curve fitting with `lmfit`.

Uses dill for IPC due to issues with pickling `lmfit` instances.
"""
import xarray as xr
from typing import Any, List, Optional
from .broadcast_common import apply_window, compile_model, unwrap_params
from dataclasses import dataclass, field
import dill

__all__ = ["MPWorker"]


@dataclass
class MPWorker:
    """Worker for performing curve fitting on a high dimensional dataset.

    Essentially represents a closure over the curve fitting parameters (the input data,
    model specification, etc.) which do not change for a batch of curve fits.

    This worker can then be treated as a function of coordinates only which:
    1. Subselects its data at those coordinates
    2. Performs the fit at those coordinates using the retained settings
    3. Returns the fit result

    There are also a few details related to serialization because we are dealing with
    IPC whenever we use multiprocessing.
    """

    data: xr.DataArray
    uncompiled_model: Any

    prefixes: Optional[List[str]]
    params: Any

    safe: bool = False
    serialize: bool = False
    weights: Optional[xr.DataArray] = None
    window: Optional[xr.DataArray] = None

    _model: Any = field(init=False)

    def __post_init__(self):
        """Indicate that the model has not been compiled yet."""
        self._model = None

    @property
    def model(self):
        """Compiles and caches the model used for curve fitting.

        Because of pickling constraints, we send model specifications
        not model instances out to workers and let them compile the model.

        This also ends up being slightly more efficient because the specification
        for a model is just references to classes in code.
        """
        if self._model is not None:
            return self._model

        self._model = compile_model(
            self.uncompiled_model, params=self.params, prefixes=self.prefixes
        )
        self._model.make_params()

        return self._model

    @property
    def fit_params(self):
        """Builds or fetches the parameter hints from closed over attributes."""
        if isinstance(self.params, (list, tuple)):
            return {}

        return self.params

    def __call__(self, cut_coords):
        """Performs a curve fit at the coordinates specified by `cut_coords`."""
        current_params = unwrap_params(self.fit_params, cut_coords)
        cut_data, original_cut_data = apply_window(self.data, cut_coords, self.window)

        if self.safe:
            cut_data = cut_data.G.drop_nan()

        weights_for = None
        if self.weights is not None:
            weights_for = self.weights.sel(**cut_coords)

        try:
            fit_result = self.model.guess_fit(cut_data, params=current_params, weights=weights_for)
        except ValueError:
            fit_result = None

        if fit_result is None:
            true_residual = None
        elif self.window is None:
            true_residual = fit_result.residual
        else:
            true_residual = original_cut_data - fit_result.eval(
                x=original_cut_data.coords[original_cut_data.dims[0]].values
            )

        if self.serialize:
            fit_result = dill.dumps(fit_result)

        return fit_result, true_residual, cut_coords
