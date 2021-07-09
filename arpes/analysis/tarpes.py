"""Very basic, generic time-resolved ARPES analysis tools."""
import numpy as np

from arpes.preparation import normalize_dim
from arpes.provenance import update_provenance
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ("find_t0", "relative_change", "normalized_relative_change")


@update_provenance("Normalized subtraction map")
def normalized_relative_change(
    data: DataType, t0=None, buffer=0.3, normalize_delay=True
) -> DataType:
    """Calculates a normalized relative Tr-ARPES change in a delay scan.

    Obtained by normalizing along the pump-probe "delay" axis and then subtracting
    the mean before t0 data and dividing by the original spectrum.

    Args:
        data: The input spectrum to be normalized. Should have a "delay" dimension.
        t0: The t0 for the input array.
        buffer: How far before t0 to select equilibrium data. Should be at least
          the temporal resolution in ps.
        normalize_delay: If true, normalizes data along the "delay" dimension.

    Returns:
        The normalized data.
    """
    spectrum = normalize_to_spectrum(data)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, "delay")
    subtracted = relative_change(spectrum, t0, buffer, normalize_delay=False)
    normalized = subtracted / spectrum
    normalized.values[np.isinf(normalized.values)] = 0
    normalized.values[np.isnan(normalized.values)] = 0
    return normalized


@update_provenance("Created simple subtraction map")
def relative_change(data: DataType, t0=None, buffer=0.3, normalize_delay=True) -> DataType:
    """Like normalized_relative_change, but only subtracts the before t0 data.

    Args:
        data: The input spectrum to be normalized. Should have a "delay" dimension.
        t0: The t0 for the input array.
        buffer: How far before t0 to select equilibrium data. Should be at least
          the temporal resolution in ps.
        normalize_delay: If true, normalizes data along the "delay" dimension.

    Returns:
        The normalized data.
    """
    spectrum = normalize_to_spectrum(data)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, "delay")

    delay_coords = spectrum.coords["delay"]
    delay_start = np.min(delay_coords)

    if t0 is None:
        t0 = spectrum.S.t0 or find_t0(spectrum)

    assert t0 - buffer > delay_start

    before_t0 = spectrum.sel(delay=slice(None, t0 - buffer))
    subtracted = spectrum - before_t0.mean("delay")
    return subtracted


def find_t0(data: DataType, e_bound=0.02) -> float:
    """Finds the effective t0 by fitting excited carriers.

    Args:
        data: A spectrum with "eV" and "delay" dimensions.
        e_bound: Lower bound on the energy to use for the fitting

    Returns:
        The delay value at the estimated t0.

    """
    spectrum = normalize_to_spectrum(data)

    assert "delay" in spectrum.dims
    assert "eV" in spectrum.dims

    sum_dims = set(spectrum.dims)
    sum_dims.remove("delay")
    sum_dims.remove("eV")

    summed = spectrum.sum(list(sum_dims)).sel(eV=slice(e_bound, None)).mean("eV")
    coord_max = summed.argmax().item()

    return summed.coords["delay"].values[coord_max]
