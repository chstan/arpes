"""Utilities related to statistical bootstraps.

It can sometimes be difficult to assess when bootstraps are appropriate, 
so make sure to consider this before you just stick a bootstrap around 
your code and stuff the resultant error bar into your papers.

This is most useful on data coming from ToF experiments, where individual electron
arrivals are counted, but even here you must be aware of tricky aspects of
the experiment: ToF-ARPES analyzers are not perfect, their efficiency can vary dramatically
across the detector due to MCP burn-in, and electron aberration and focusing
must be considered.
"""

import copy
from dataclasses import dataclass

import functools
import random

import scipy.stats
from typing import Any, Callable, Dict, List, Optional, Set, Union
import numpy as np
from tqdm import tqdm_notebook

import xarray as xr
from arpes.analysis.sarpes import to_intensity_polarization
from arpes.provenance import update_provenance
from arpes.typing import DataType
from arpes.utilities import lift_dataarray_to_generic
from arpes.utilities.normalize import normalize_to_spectrum
from arpes.utilities.region import normalize_region

__all__ = (
    "bootstrap",
    "estimate_prior_adjustment",
    "resample_true_counts",
    "bootstrap_counts",
    "bootstrap_intensity_polarization",
    "Normal",
    "propagate_errors",
)


@update_provenance("Estimate prior")
def estimate_prior_adjustment(data: DataType, region: Union[Dict[str, Any], str] = None) -> float:
    r"""Estimates the parameters of a distribution generating the intensity histogram of pixels in a spectrum.

    In a perfectly linear, single-electron
    single-count detector, this would be a poisson distribution with
    \lambda=mean(counts) over the window. Despite this, we can estimate \lambda
    phenomenologically and verify that a Poisson distribution provides a good
    prior for the data, allowing us to perform statistical bootstrapping.

    You should use this with a spectrum that has uniform intensity, i.e. with a
    copper reference or similar.

    Args:
        data: The input spectrum.
        region: The region which should be used for the estimate.

    Returns:
        sigma / mu, the adjustment factor for the Poisson distribution
    """
    data = normalize_to_spectrum(data)

    if region is None:
        region = "copper_prior"

    region = normalize_region(region)

    if "cycle" in data.dims:
        data = data.sum("cycle")

    data = data.S.zero_spectrometer_edges().S.region_sel(region)
    values = data.values.ravel()
    values = values[np.where(values)]
    return np.std(values) / np.mean(values)


@update_provenance("Resample cycle dimension")
@lift_dataarray_to_generic
def resample_cycle(data: xr.DataArray, **kwargs) -> xr.DataArray:
    """Perform a non-parametric bootstrap using a cycle coordinate for statistically independent observations.

    Args:
        data: The input data.
        kwargs: Unused

    Returns:
        Resampled data with selections from the cycle axis.
    """
    n_cycles = len(data.cycle)
    which = [random.randint(0, n_cycles - 1) for _ in range(n_cycles)]

    resampled = data.isel(cycle=which).sum("cycle", keep_attrs=True)

    if "id" in resampled.attrs:
        del resampled.attrs["id"]

    return resampled


@update_provenance("Resample with prior adjustment")
@lift_dataarray_to_generic
def resample(data: xr.DataArray, prior_adjustment=1, **kwargs):
    resampled = xr.DataArray(
        np.random.poisson(lam=data.values * prior_adjustment, size=data.values.shape),
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs,
    )

    if "id" in resampled.attrs:
        del resampled.attrs["id"]

    return resampled


@update_provenance("Resample electron-counted data")
@lift_dataarray_to_generic
def resample_true_counts(data: xr.DataArray) -> xr.DataArray:
    """Resamples histogrammed data where each count represents an actual electron.

    Args:
        data: Input data representing actual electron counts from a time of flight
              system or delay line.

    Returns:
        Poisson resampled data.
    """
    resampled = xr.DataArray(
        np.random.poisson(lam=data.values, size=data.values.shape),
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs,
    )

    if "id" in resampled.attrs:
        del resampled.attrs["id"]

    return resampled


@update_provenance("Bootstrap true electron counts")
@lift_dataarray_to_generic
def bootstrap_counts(data: DataType, N=1000, name=None) -> xr.Dataset:
    """Performs a parametric bootstrap assuming recorded data are electron counts.

    Parametric bootstrap for the number of counts in each detector channel for a
    time of flight/DLD detector, where each count represents an actual particle.

    This function also introspects the data passed to determine whether there is a
    spin degree of freedom, and will bootstrap appropriately.

    Currently we build all the samples at once instead of using a rolling algorithm.

    Arguments:
        data: The input spectrum.
        N: The number of samples to draw.
        name: The name of the subarray which represents counts to resample. E.g. "up_spectrum"

    Returns:
        A `xr.Dataset` which has the mean and standard error for the resampled named array.
    """
    assert data.name is not None or name is not None
    name = data.name if data.name is not None else name

    desc_fragment = " {}".format(name)

    resampled_sets = []
    for _ in tqdm_notebook(range(N), desc="Resampling{}...".format(desc_fragment)):
        resampled_sets.append(resample_true_counts(data))

    resampled_arr = np.stack([s.values for s in resampled_sets], axis=0)
    std = np.std(resampled_arr, axis=0)
    std = xr.DataArray(std, data.coords, data.dims)
    mean = np.mean(resampled_arr, axis=0)
    mean = xr.DataArray(mean, data.coords, data.dims)

    data_vars = {}
    data_vars[name] = mean
    data_vars[name + "_std"] = std

    return xr.Dataset(data_vars=data_vars, coords=data.coords, attrs=data.attrs.copy())


class Distribution:
    DEFAULT_N_SAMPLES = 1000

    def draw_samples(self, n_samples=DEFAULT_N_SAMPLES):
        """Draws samples from this distribution."""
        raise NotImplementedError


@dataclass
class Normal(Distribution):
    """Represents a Gaussian distribution.

    Attributes:
        center: The center/mu parameter for the distribution.
        stderr: The standard error for the distribution.
    """

    center: float
    stderr: float

    def draw_samples(self, n_samples=Distribution.DEFAULT_N_SAMPLES):
        """Draws samples from this distribution."""
        return scipy.stats.norm.rvs(self.center, scale=self.stderr, size=n_samples)

    @classmethod
    def from_param(cls, model_param):
        """Generates a Normal from an `lmfit.Parameter`."""
        return cls(center=model_param.value, stderr=model_param.stderr)


def propagate_errors(f) -> Callable:
    """A decorator which provides transparent propagation of statistical errors.

    The way that this is accomodated is that the inner function is turned into one which
    operates over distributions. Errors are calculated empirically by sampling
    over trials drawn from these distributions.

    CAVEAT EMPTOR: Arguments are assumed to be uncorrelated.

    Args:
        f: The inner function

    Returns:
        The wrapped function.
    """

    @functools.wraps(f)
    def operates_on_distributions(*args, **kwargs):
        exclude = set(
            [i for i, arg in enumerate(args) if not isinstance(arg, Distribution)]
            + [k for k, arg in kwargs.items() if not isinstance(arg, Distribution)]
        )

        if len(exclude) == len(args) + len(kwargs):
            # short circuit if no bootstrapping is required to be nice to the user
            return f(*args, **kwargs)

        vec_f = np.vectorize(f, excluded=exclude)
        res = vec_f(
            *[a.draw_samples() if isinstance(a, Distribution) else a for a in args],
            **{k: v.draw_samples() if isinstance(v, Distribution) else v for k, v in kwargs.items()}
        )

        try:
            print(scipy.stats.describe(res))
        except:
            pass

        return res

    return operates_on_distributions


@update_provenance("Bootstrap spin detector polarization and intensity")
def bootstrap_intensity_polarization(data: xr.Dataset, N: int = 100) -> xr.Dataset:
    """Builds an estimate of the intensity and polarization from spin-data.

    Uses the parametric bootstrap to get uncertainties on the intensity and polarization of ToF-SARPES data.

    Args:
        data: Input spectrum for resampling.
        N: The number of samples to draw.

    Returns:
        Resampled data after conversion to intensity and polarization.
    """
    bootstrapped_polarization = bootstrap(to_intensity_polarization)
    return bootstrapped_polarization(data, N=N)


def bootstrap(
    fn: Callable,
    skip: Optional[Union[Set[int], List[int]]] = None,
    resample_method: Optional[str] = None,
) -> Callable:
    """Produces function which performs a bootstrap of an arbitrary function by sampling.

    This is a functor which takes a function operating on plain data and produces one which
    internally bootstraps over counts on the input data.

    Args:
        fn: The function to be bootstrapped.
        skip: Which arguments to leave alone. Defaults to None.
        resample_method: How the resampling should be performed. See `resample` and `resample_cycle`. Defaults to None.

    Returns:
        A function which vectorizes the ouptut of the input function `fn` over samples.
    """
    if skip is None:
        skip = []

    skip = set(skip)

    if resample_method is None:
        resample_fn = resample
    elif resample_method == "cycle":
        resample_fn = resample_cycle

    def bootstrapped(*args, N=20, prior_adjustment=1, **kwargs):
        # examine args to determine which to resample
        resample_indices = [
            i
            for i, arg in enumerate(args)
            if isinstance(arg, (xr.DataArray, xr.Dataset)) and i not in skip
        ]
        data_is_arraylike = False

        runs = []

        def get_label(i):
            if isinstance(args[i], xr.Dataset):
                return "xr.Dataset: [{}]".format(", ".join(args[i].data_vars.keys()))
            if args[i].name:
                return args[i].name

            try:
                return args[i].attrs["id"]
            except KeyError:
                return "Label-less DataArray"

        print("Resampling args: {}".format(",".join([get_label(i) for i in resample_indices])))

        # examine kwargs to determine which to resample
        resample_kwargs = [
            k for k, v in kwargs.items() if isinstance(v, xr.DataArray) and k not in skip
        ]
        print("Resampling kwargs: {}".format(",".join(resample_kwargs)))

        print(
            "Fair warning 1: Make sure you understand whether it is appropriate to resample your data."
        )
        print(
            "Fair warning 2: Ensure that the data to resample is in a DataArray and not a Dataset"
        )

        for _ in tqdm_notebook(range(N), desc="Resampling..."):
            new_args = list(args)
            new_kwargs = copy.copy(kwargs)
            for i in resample_indices:
                new_args[i] = resample_fn(args[i], prior_adjustment=prior_adjustment)
            for k in resample_kwargs:
                new_kwargs[k] = resample_fn(kwargs[k], prior_adjustment=prior_adjustment)

            run = fn(*new_args, **new_kwargs)
            if isinstance(run, (xr.DataArray, xr.Dataset)):
                data_is_arraylike = True
            runs.append(run)

        if data_is_arraylike:
            for i, run in enumerate(runs):
                run = run.assign_coords(bootstrap=i)

            return xr.concat(runs, dim="bootstrap")

        return runs

    return functools.wraps(fn)(bootstrapped)
