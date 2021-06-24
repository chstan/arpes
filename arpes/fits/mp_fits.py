import xarray as xr
from typing import Any, List, Optional
from .broadcast_common import apply_window, compile_model, unwrap_params
from dataclasses import dataclass, field
import dill

__all__ = ["mp_fit"]


@dataclass
class MPWorker:
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
        self._model = None

    @property
    def model(self):
        if self._model is not None:
            return self._model

        self._model = compile_model(
            self.uncompiled_model, params=self.params, prefixes=self.prefixes
        )
        self._model.make_params()

        return self._model

    @property
    def fit_params(self):
        if isinstance(self.params, (list, tuple)):
            return {}

        return self.params

    def __call__(self, cut_coords):
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
