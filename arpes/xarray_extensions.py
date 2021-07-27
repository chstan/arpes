"""Establishes the PyARPES data model by extending the `xarray` types.

This is another core part of PyARPES. It provides a lot of extensions to
what comes out of the box in xarray. Some of these are useful generics,
generally on the .T extension, others collect and manipulate metadata,
interface with plotting routines, provide functional programming utilities,
etc.

If `f` is an ARPES spectrum, then `f.S` should provide a nice representation of your data
in a Jupyter cell. This is a complement to the text based approach that merely printing `f`
offers. Note, as of PyARPES v3.x.y, the xarray version has been bumped and this representation
is no longer necessary as one is provided upstream.

The main accessors are .S, .G, .X. and .F.

The `.S` accessor:
    The `.S` accessor contains functionality related to spectroscopy. Utilities 
    which only make sense in this context should be placed here, while more generic
    tools should be placed elsewhere.

The `.G.` accessor:
    This a general purpose collection of tools which exists to provide conveniences
    over what already exists in the xarray data model. As an example, there are 
    various tools for simultaneous iteration of data and coordinates here, as well as 
    for vectorized application of functions to data or coordinates.

The `.X` accessor: 
    This is an accessor which contains tools related to selecting and subselecting 
    data. The two most notable tools here are `.X.first_exceeding` which is very useful
    for initializing curve fits and `.X.max_in_window` which is useful for refining 
    these initial parameter choices.

The `.F.` accessor:
    This is an accessor which contains tools related to interpreting curve fitting
    results. In particular there are utilities for vectorized extraction of parameters,
    for plotting several curve fits, or for selecting "worst" or "best" fits according
    to some measure.
"""

import pandas as pd
import lmfit
import arpes
import contextlib
import collections
import copy
import itertools
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scipy import ndimage as ndi

import arpes.constants
import arpes.plotting as plotting

from arpes.analysis.general import rebin
from arpes.analysis.band_analysis_utils import param_getter, param_stderr_getter
from arpes.models.band import MultifitBand
from arpes.plotting.utils import fancy_labels, remove_colorbars
from arpes.plotting.parameter import plot_parameter
from arpes.typing import DataType, DTypeLike
from arpes.utilities import apply_dataarray
from arpes.utilities.collections import MappableDict
from arpes.utilities.conversion import slice_along_path
import arpes.utilities.math
from arpes.utilities.region import DesignatedRegions, normalize_region
from arpes.utilities.xarray import unwrap_xarray_item, unwrap_xarray_dict


__all__ = ["ARPESDataArrayAccessor", "ARPESDatasetAccessor", "ARPESFitToolsAccessor"]


def _iter_groups(grouped: Dict[str, Any]) -> Iterator[Any]:
    """Iterates through a flattened sequence.

    Sequentially yields keys and values from each sequence associated with a key.
    If a key "k" is associated to a value "v0" which is not iterable, it will be emitted as a
    single pair [k, v0].

    Otherwise, one pair is yielded for every item in the associated value.
    """
    for k, value_or_list in grouped.items():
        try:
            for list_item in value_or_list:
                yield k, list_item
        except TypeError:
            yield k, value_or_list


class ARPESAccessorBase:
    """Base class for the xarray extensions in PyARPES."""

    def along(self, directions, **kwargs):
        return slice_along_path(self._obj, directions, **kwargs)

    def find(self, name):
        return [n for n in dir(self) if name in n]

    @property
    def sherman_function(self):
        for option in ["sherman", "sherman_function", "SHERMAN"]:
            if option in self._obj.attrs:
                return self._obj.attrs[option]

        raise ValueError("No Sherman function could be found on the data. Is this a spin dataset?")

    @property
    def experimental_conditions(self):
        try:
            temp = self.temp
        except AttributeError:
            temp = None

        return {
            "hv": self.hv,
            "polarization": self.polarization,
            "temp": temp,
        }

    @property
    def polarization(self):
        if "epu_pol" in self._obj.attrs:
            # merlin: TODO normalize these
            # check and complete
            try:
                return {
                    0: "p",
                    1: "rc",
                    2: "s",
                }.get(int(self._obj.attrs["epu_pol"]))
            except ValueError:
                return self._obj.attrs["epu_pol"]

        return None

    @property
    def is_subtracted(self):
        if self._obj.attrs.get("subtracted"):
            return True

        if isinstance(self._obj, xr.DataArray):
            # if at least 5% of the values are < 0 we should consider the data
            # to be best represented by a coolwarm map
            return (((self._obj < 0) * 1).mean() > 0.05).item()

    @property
    def is_spatial(self) -> bool:
        """Infers whether a given scan has real-space dimensions (SPEM or u/nARPES).

        Returns:
            True if the data is explicltly a "ucut" or "spem" or contains "X", "Y", or "Z", dimensions.
            False otherwise.
        """
        if self.spectrum_type in {"ucut", "spem"}:
            return True

        return any(d in {"X", "Y", "Z"} for d in self._obj.dims)

    @property
    def is_kspace(self) -> bool:
        """Infers whether the scan is k-space converted or not.

        Because of the way this is defined, it will return
        true for XPS spectra, which I suppose is true but trivially.

        Returns:
            True if the data is k-space converted.
            False otherwise.
        """
        return not any(d in {"phi", "theta", "beta", "angle"} for d in self._obj.dims)

    @property
    def is_slit_vertical(self) -> bool:
        """Whether the data was taken on an analyzer with vertical slit.

        Caveat emptor: this assumes that the alpha coordinate is not some intermediate value.

        Returns:
            True if the alpha value is consistent with a vertical slit analyzer.
            False otherwise.
        """
        return np.abs(self.lookup_offset_coord("alpha") - np.pi / 2) < (np.pi / 180)

    @property
    def endstation(self) -> str:
        """Alias for the location attribute used to load the data.

        Returns:
            The name of loader/location which was used to load data.
        """
        return self._obj.attrs["location"]

    def with_values(self, new_values: np.ndarray) -> xr.DataArray:
        """Copy with new array values.

        Easy way of creating a DataArray that has the same shape as the calling object but data populated
        from the array `new_values`

        Args:
            new_values: The new values which should be used for the data.

        Returns:
            A copy of the data with new values but identical dimensions, coordinates, and attrs.
        """
        return xr.DataArray(
            new_values.reshape(self._obj.values.shape),
            coords=self._obj.coords,
            dims=self._obj.dims,
            attrs=self._obj.attrs,
        )

    def with_standard_coords(self):
        obj = self._obj

        collected_renamings = {}
        coord_names = {
            "pixel",
            "eV",
            "phi",
        }
        for coord_name in coord_names:
            clarified = [name for name in obj.coords.keys() if (coord_name + "-") in name]
            assert len(clarified) < 2

            if clarified:
                collected_renamings[clarified[0]] = coord_name

        return obj.rename(collected_renamings)

    @property
    def logical_offsets(self):
        if "long_x" not in self._obj.coords:
            raise ValueError(
                "Logical offsets can currently only be "
                "accessed for hierarchical motor systems like nanoARPES."
            )

        return MappableDict(
            unwrap_xarray_dict(
                {
                    "x": self._obj.coords["long_x"] - self._obj.coords["physical_long_x"],
                    "y": self._obj.coords["long_y"] - self._obj.coords["physical_long_y"],
                    "z": self._obj.coords["long_z"] - self._obj.coords["physical_long_z"],
                }
            )
        )

    @property
    def hv(self):
        if "hv" in self._obj.coords:
            value = float(self._obj.coords["hv"])
            if not np.isnan(value):
                return value

        if "hv" in self._obj.attrs:
            value = float(self._obj.attrs["hv"])
            if not np.isnan(value):
                return value

        if "location" in self._obj.attrs:
            if self._obj.attrs["location"] == "ALG-MC":
                return 5.93

        return None

    def fetch_ref_attrs(self):
        if "ref_attrs" in self._obj.attrs:
            return self._obj.attrs

        raise NotImplementedError

    @property
    def scan_type(self):
        return self._obj.attrs.get("daq_type")

    @property
    def spectrum_type(self):
        if "spectrum_type" in self._obj.attrs and self._obj.attrs["spectrum_type"]:
            return self._obj.attrs["spectrum_type"]

        dim_types = {
            ("eV",): "xps_spectrum",
            ("eV", "phi"): "spectrum",
            # this should check whether the other angular axis perpendicular to scan axis?
            ("eV", "phi", "beta"): "map",
            ("eV", "phi", "theta"): "map",
            ("eV", "hv", "phi"): "hv_map",
            # kspace
            ("eV", "kp"): "spectrum",
            ("eV", "kx", "ky"): "map",
            ("eV", "kp", "kz"): "hv_map",
        }

        dims = tuple(sorted(list(self._obj.dims)))

        return dim_types.get(dims)

    @property
    def is_differentiated(self):
        history = self.short_history()
        return "dn_along_axis" in history or "curvature" in history

    def transpose_to_front(self, dim):
        dims = list(self._obj.dims)
        assert dim in dims
        dims.remove(dim)
        return self._obj.transpose(*([dim] + dims))

    def transpose_to_back(self, dim):
        dims = list(self._obj.dims)
        assert dim in dims
        dims.remove(dim)
        return self._obj.transpose(*(dims + [dim]))

    def select_around_data(
        self,
        points: Union[Dict[str, Any], xr.Dataset],
        radius: Dict[str, float] = None,
        fast: bool = False,
        safe: bool = True,
        mode: str = "sum",
        **kwargs,
    ):
        """Performs a binned selection around a point or points.

        Can be used to perform a selection along one axis as a function of another, integrating a region
        in the other dimensions.

        Example:
            As an example, suppose we have a dataset with dimensions ('eV', 'kp', 'T',)
            and we also by fitting determined the Fermi momentum as a function of T, kp_F('T'), stored in the
            dataarray kFs. Then we could select momentum integrated EDCs in a small window around the fermi momentum
            for each temperature by using

            >>> edcs_at_fermi_momentum = full_data.S.select_around_data({'kp': kFs}, radius={'kp': 0.04}, fast=True)  # doctest: +SKIP

            The resulting data will be EDCs for each T, in a region of radius 0.04 inverse angstroms around the
            Fermi momentum.

        Args:
            points: The set of points where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a standard sized
                    selection will be made as a compromise.
            fast: If true, uses a rectangular rather than a circular region for selectioIf true, uses a
                  rectangular rather than a circular region for selection.
            safe: If true, infills radii with default values. Defaults to `True`.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"
            kwargs: Can be used to pass radii parameters by keyword with `_r` postfix.

        Returns:
            The binned selection around the desired point or points.
        """
        if isinstance(self._obj, xr.Dataset):
            raise TypeError("Cannot use select_around on Datasets only DataArrays!")

        if mode not in {"sum", "mean"}:
            raise ValueError("mode parameter should be either sum or mean.")

        if isinstance(
            points,
            (
                tuple,
                list,
            ),
        ):
            warnings.warn("Dangerous iterable points argument to `select_around`")
            points = dict(zip(points, self._obj.dims))
        if isinstance(points, xr.Dataset):
            points = {k: points[k].item() for k in points.data_vars}

        default_radii = {
            "kp": 0.02,
            "kz": 0.05,
            "phi": 0.02,
            "beta": 0.02,
            "theta": 0.02,
            "eV": 0.05,
            "delay": 0.2,
            "T": 2,
            "temp": 2,
        }

        unspecified = 0.1

        if isinstance(radius, float):
            radius = {d: radius for d in points.keys()}
        else:
            collected_terms = set("{}_r".format(k) for k in points.keys()).intersection(
                set(kwargs.keys())
            )
            if collected_terms:
                radius = {
                    d: kwargs.get("{}_r".format(d), default_radii.get(d, unspecified))
                    for d in points.keys()
                }
            elif radius is None:
                radius = {d: default_radii.get(d, unspecified) for d in points.keys()}

        assert isinstance(radius, dict)
        radius = {d: radius.get(d, default_radii.get(d, unspecified)) for d in points.keys()}

        along_dims = list(points.values())[0].dims
        selected_dims = list(points.keys())

        stride = self._obj.G.stride(generic_dim_names=False)

        new_dim_order = [d for d in self._obj.dims if d not in along_dims] + list(along_dims)

        data_for = self._obj.transpose(*new_dim_order)
        new_data = data_for.sum(selected_dims, keep_attrs=True)
        for coord, value in data_for.G.iterate_axis(along_dims):
            nearest_sel_params = {}
            if safe:
                for d, v in radius.items():
                    if v < stride[d]:
                        nearest_sel_params[d] = points[d].sel(**coord)

                radius = {d: v for d, v in radius.items() if d not in nearest_sel_params}

            if fast:
                selection_slices = {
                    d: slice(
                        points[d].sel(**coord) - radius[d],
                        points[d].sel(**coord) + radius[d],
                    )
                    for d in points.keys()
                    if d in radius
                }
                selected = value.sel(**selection_slices)
            else:
                raise NotImplementedError

            if nearest_sel_params:
                selected = selected.sel(**nearest_sel_params, method="nearest")

            for d in nearest_sel_params:
                # need to remove the extra dims from coords
                del selected.coords[d]

            if mode == "sum":
                new_data.loc[coord] = selected.sum(list(radius.keys())).values
            elif mode == "mean":
                new_data.loc[coord] = selected.mean(list(radius.keys())).values

        return new_data

    def select_around(
        self,
        point: Union[Dict[str, Any], xr.Dataset],
        radius: Dict[str, float] = None,
        fast: bool = False,
        safe: bool = True,
        mode: str = "sum",
        **kwargs,
    ) -> xr.DataArray:
        """Selects and integrates a region around a one dimensional point.

        This method is useful to do a small region integration, especially around
        points on a path of a k-point of interest. See also the companion method `select_around_data`.

        If the fast flag is set, we will use the Manhattan norm, i.e. sum over square regions
        rather than ellipsoids, as this is less costly.

        If radii are not set, or provided through kwargs as 'eV_r' or 'phi_r' for instance,
        then we will try to use reasonable default values; buyer beware.

        Args:
            point: The points where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a standard sized
                    selection will be made as a compromise.
            fast: If true, uses a rectangular rather than a circular region for selectioIf true, uses a
                  rectangular rather than a circular region for selection.
            safe: If true, infills radii with default values. Defaults to `True`.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"
            kwargs: Can be used to pass radii parameters by keyword with `_r` postfix.

        Returns:
            The binned selection around the desired point or points.
        """
        if isinstance(self._obj, xr.Dataset):
            raise TypeError("Cannot use select_around on Datasets only DataArrays!")

        if mode not in {"sum", "mean"}:
            raise ValueError("mode parameter should be either sum or mean.")

        if isinstance(
            point,
            (
                tuple,
                list,
            ),
        ):
            warnings.warn("Dangerous iterable point argument to `select_around`")
            point = dict(zip(point, self._obj.dims))
        if isinstance(point, xr.Dataset):
            point = {k: point[k].item() for k in point.data_vars}

        default_radii = {
            "kp": 0.02,
            "kz": 0.05,
            "phi": 0.02,
            "beta": 0.02,
            "theta": 0.02,
            "eV": 0.05,
            "delay": 0.2,
            "T": 2,
        }

        unspecified = 0.1

        if isinstance(radius, float):
            radius = {d: radius for d in point.keys()}
        else:
            collected_terms = set("{}_r".format(k) for k in point.keys()).intersection(
                set(kwargs.keys())
            )
            if collected_terms:
                radius = {
                    d: kwargs.get("{}_r".format(d), default_radii.get(d, unspecified))
                    for d in point.keys()
                }
            elif radius is None:
                radius = {d: default_radii.get(d, unspecified) for d in point.keys()}

        assert isinstance(radius, dict)
        radius = {d: radius.get(d, default_radii.get(d, unspecified)) for d in point.keys()}

        # make sure we are taking at least one pixel along each
        nearest_sel_params = {}
        if safe:
            stride = self._obj.G.stride(generic_dim_names=False)
            for d, v in radius.items():
                if v < stride[d]:
                    nearest_sel_params[d] = point[d]

            radius = {d: v for d, v in radius.items() if d not in nearest_sel_params}

        if fast:
            selection_slices = {
                d: slice(point[d] - radius[d], point[d] + radius[d])
                for d in point.keys()
                if d in radius
            }
            selected = self._obj.sel(**selection_slices)
        else:
            # selected = self._obj
            raise NotImplementedError

        if nearest_sel_params:
            selected = selected.sel(**nearest_sel_params, method="nearest")

        for d in nearest_sel_params:
            # need to remove the extra dims from coords
            del selected.coords[d]

        if mode == "sum":
            return selected.sum(list(radius.keys()))
        elif mode == "mean":
            return selected.mean(list(radius.keys()))

    def short_history(self, key="by"):
        return [h["record"][key] if isinstance(h, dict) else h for h in self.history]

    def _calculate_symmetry_points(
        self,
        symmetry_points,
        projection_distance=0.05,
        include_projected=True,
        epsilon=0.01,
    ):
        # For each symmetry point, we need to determine if it is projected or not
        # if it is projected, we need to calculate its projected coordinates
        points = collections.defaultdict(list)
        projected_points = collections.defaultdict(list)

        fixed_coords = {k: v for k, v in self._obj.coords.items() if k not in self._obj.indexes}
        index_coords = self._obj.indexes

        for point, locations in symmetry_points.items():
            if not isinstance(locations, list):
                locations = [locations]

            for location in locations:
                # determine whether the location needs to be projected
                projected = False
                skip = False
                for axis, value in location.items():
                    if axis in fixed_coords and np.abs(value - fixed_coords[axis]) > epsilon:
                        projected = True
                    if axis not in fixed_coords and axis not in index_coords:
                        # cannot even hope to do anything here, we don't have enough info
                        skip = True

                if skip:
                    continue

                new_location = location.copy()
                if projected:
                    # Go and do the projection, for now we will assume we just get it by
                    # replacing the value of the mismatched coordinates.

                    # This does not work if the coordinate system is not orthogonal
                    for axis, v in location.items():
                        if axis in fixed_coords:
                            new_location = fixed_coords[axis]

                    projected_points[point].append(location)
                else:
                    points[point].append(location)

        return points, projected_points

    def symmetry_points(self, raw=False, **kwargs):
        try:
            symmetry_points = self.fetch_ref_attrs().get("symmetry_points", {})
        except:
            symmetry_points = {}
        our_symmetry_points = self._obj.attrs.get("symmetry_points", {})

        symmetry_points = copy.deepcopy(symmetry_points)
        symmetry_points.update(our_symmetry_points)

        if raw:
            return symmetry_points

        return self._calculate_symmetry_points(symmetry_points, **kwargs)

    @property
    def iter_own_symmetry_points(self):
        sym_points, _ = self.symmetry_points()
        return _iter_groups(sym_points)

    @property
    def iter_projected_symmetry_points(self):
        _, sym_points = self.symmetry_points()
        return _iter_groups(sym_points)

    @property
    def iter_symmetry_points(self):
        for sym_point in self.iter_own_symmetry_points:
            yield sym_point
        for sym_point in self.iter_projected_symmetry_points:
            yield sym_point

    @property
    def history(self):
        provenance = self._obj.attrs.get("provenance", None)

        def unlayer(prov):
            if prov is None:
                return [], None
            if isinstance(prov, str):
                return [prov], None
            first_layer = copy.copy(prov)

            rest = first_layer.pop("parents_provenance", None)
            if rest is None:
                rest = first_layer.pop("parents_provanence", None)
            if isinstance(rest, list):
                warnings.warn(
                    "Encountered multiple parents in history extraction, "
                    "throwing away all but the first."
                )
                if rest:
                    rest = rest[0]
                else:
                    rest = None

            return [first_layer], rest

        def _unwrap_provenance(prov):
            if prov is None:
                return []

            first, rest = unlayer(prov)

            return first + _unwrap_provenance(rest)

        return _unwrap_provenance(provenance)

    @property
    def spectrometer(self):
        ds = self._obj
        spectrometers = {
            "SToF": arpes.constants.SPECTROMETER_SPIN_TOF,
            "ToF": arpes.constants.SPECTROMETER_STRAIGHT_TOF,
            "DLD": arpes.constants.SPECTROMETER_DLD,
            "BL7": arpes.constants.SPECTROMETER_BL7,
            "ANTARES": arpes.constants.SPECTROMETER_ANTARES,
        }

        if "spectrometer_name" in ds.attrs:
            return spectrometers.get(ds.attrs["spectrometer_name"])

        if isinstance(ds, xr.Dataset):
            if "up" in ds.data_vars or ds.attrs.get("18  MCP3") == 0:
                return spectrometers["SToF"]
        elif isinstance(ds, xr.DataArray):
            if ds.name == "up" or ds.attrs.get("18  MCP3") == 0:
                return spectrometers["SToF"]

        if "location" in ds.attrs:
            return {
                "ALG-MC": arpes.constants.SPECTROMETER_MC,
                "BL403": arpes.constants.SPECTROMETER_BL4,
                "ALG-SToF": arpes.constants.SPECTROMETER_STRAIGHT_TOF,
                "Kaindl": arpes.constants.SPECTROMETER_KAINDL,
                "BL7": arpes.constants.SPECTROMETER_BL7,
                "ANTARES": arpes.constants.SPECTROMETER_ANTARES,
            }.get(ds.attrs["location"])

        try:
            return spectrometers[ds.attrs["spectrometer_name"]]
        except KeyError:
            return {}

    @property
    def dshape(self):
        arr = self._obj
        return dict(zip(arr.dims, arr.shape))

    @property
    def original_id(self):
        history = self.history
        if len(history) >= 3:
            first_modification = history[-3]
            return first_modification["parent_id"]

        return self._obj.attrs["id"]

    @property
    def original_parent_scan_name(self):
        try:
            history = self.history
            if len(history) >= 3:
                first_modification = history[-3]
                df = self._obj.attrs["df"]
                return df[df.id == first_modification["parent_id"]].index[0]
        except:
            pass
        return ""

    @property
    def scan_row(self):
        df = self._obj.attrs["df"]
        sdf = df[df.path == self._obj.attrs["file"]]
        return list(sdf.iterrows())[0]

    @property
    def df_index(self):
        return self.scan_row[0]

    @property
    def df_after(self):
        return self._obj.attrs["df"][self._obj.attrs["df"].index > self.df_index]

    def df_until_type(self, df=None, spectrum_type=None):
        if df is None:
            df = self.df_after

        if spectrum_type is None:
            spectrum_type = (self.spectrum_type,)

        if isinstance(spectrum_type, str):
            spectrum_type = (spectrum_type,)

        try:
            indices = [df[df["spectrum_type"].eq(s)] for s in spectrum_type]
            indices = [d.index[0] for d in indices if not d.empty]

            if not indices:
                raise IndexError()

            min_index = min(indices)
            return df[df.index < min_index]
        except IndexError:
            # nothing
            return df

    @property
    def scan_name(self):
        for option in ["scan", "file"]:
            if option in self._obj.attrs:
                return self._obj.attrs[option]

        id = self._obj.attrs.get("id")

        if id is None:
            return "No ID"

        try:
            df = self._obj.attrs["df"]
            return df[df.id == id].index[0]
        except (IndexError, KeyError, AttributeError):
            # data is probably not raw data
            return self.original_parent_scan_name

    @property
    def label(self):
        return str(self._obj.attrs.get("description", self.scan_name))

    @property
    def t0(self):
        if "t0" in self._obj.attrs:
            value = float(self._obj.attrs["t0"])
            if not np.isnan(value):
                return value

        if "T0_ps" in self._obj.attrs:
            value = float(self._obj.attrs["T0_ps"])
            if not np.isnan(value):
                return value

        return None

    @contextlib.contextmanager
    def with_rotation_offset(self, offset: float):
        """Temporarily rotates the chi_offset by `offset`."""
        old_chi_offset = self.offsets.get("chi", 0)
        self.apply_offsets({"chi": old_chi_offset + offset})

        yield old_chi_offset + offset

        self.apply_offsets({"chi": old_chi_offset})

    def apply_offsets(self, offsets):
        for k, v in offsets.items():
            self._obj.attrs["{}_offset".format(k)] = v

    @property
    def offsets(self):
        return {
            c: self.lookup_offset(c) for c in self._obj.coords if f"{c}_offset" in self._obj.attrs
        }

    def lookup_offset_coord(self, name):
        return self.lookup_coord(name) - self.lookup_offset(name)

    def lookup_coord(self, name):
        if name in self._obj.coords:
            return unwrap_xarray_item(self._obj.coords[name])

        if name in self._obj.attrs:
            return unwrap_xarray_item(self._obj.attrs[name])

        raise ValueError("Could not find coordinate {}.".format(name))

    def lookup_offset(self, attr_name):
        symmetry_points = self.symmetry_points(raw=True)
        if "G" in symmetry_points:
            gamma_point = symmetry_points["G"]
            if attr_name in gamma_point:
                return unwrap_xarray_item(gamma_point[attr_name])

        offset_name = attr_name + "_offset"
        if offset_name in self._obj.attrs:
            return unwrap_xarray_item(self._obj.attrs[offset_name])

        return unwrap_xarray_item(self._obj.attrs.get("data_preparation", {}).get(offset_name, 0))

    @property
    def beta_offset(self):
        return self.lookup_offset("beta")

    @property
    def psi_offset(self):
        return self.lookup_offset("psi")

    @property
    def theta_offset(self):
        return self.lookup_offset("theta")

    @property
    def phi_offset(self):
        return self.lookup_offset("phi")

    @property
    def chi_offset(self):
        return self.lookup_offset("chi")

    @property
    def work_function(self) -> float:
        """Provides the work function, if present in metadata.

        Otherwise, uses something approximate.
        """
        if "sample_workfunction" in self._obj.attrs:
            return self._obj.attrs["sample_workfunction"]

        return 4.3

    @property
    def inner_potential(self) -> float:
        """Provides the inner potential, if present in metadata.

        Otherwise, 10eV is assumed.
        """
        if "inner_potential" in self._obj.attrs:
            return self._obj.attrs["inner_potential"]

        return 10

    def find_spectrum_energy_edges(self, indices=False):
        energy_marginal = self._obj.sum([d for d in self._obj.dims if d not in ["eV"]])

        embed_size = 20
        embedded = np.ndarray(shape=[embed_size] + list(energy_marginal.values.shape))
        embedded[:] = energy_marginal.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        # try to avoid dependency conflict with numpy v0.16
        from skimage import feature  # pylint: disable=import-error

        edges = (
            feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True, low_threshold=0.3) * 1
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.G.stride(generic_dim_names=False)
        return edges * delta["eV"] + self._obj.coords["eV"].values[0]

    def find_spectrum_angular_edges_full(self, indices=False):
        # as a first pass, we need to find the bottom of the spectrum, we will use this
        # to select the active region and then to rebin into course steps in energy from 0
        # down to this region
        # we will then find the appropriate edge for each slice, and do a fit to the edge locations

        energy_edge = self.find_spectrum_energy_edges()
        low_edge = np.min(energy_edge) + 0.05
        high_edge = np.max(energy_edge) - 0.05

        if high_edge - low_edge < 0.15:
            # Doesn't look like the automatic inference of the energy edge was valid
            high_edge = 0
            low_edge = np.min(self._obj.coords["eV"].values)

        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        energy_cut = self._obj.sel(eV=slice(low_edge, high_edge)).S.sum_other(["eV", angular_dim])

        n_cuts = int(np.ceil(high_edge - low_edge / 0.05))
        new_shape = {"eV": n_cuts}
        new_shape[angular_dim] = len(energy_cut.coords[angular_dim].values)
        rebinned = rebin(energy_cut, shape=new_shape)

        embed_size = 20
        embedded = np.ndarray(shape=[embed_size] + [len(rebinned.coords[angular_dim].values)])
        low_edges = []
        high_edges = []
        for e_cut in rebinned.coords["eV"].values:
            e_slice = rebinned.sel(eV=e_cut)
            values = e_slice.values
            values[values > np.mean(values)] = np.mean(values)
            embedded[:] = values
            embedded = ndi.gaussian_filter(embedded, embed_size / 1.5)

            # try to avoid dependency conflict with numpy v0.16
            from skimage import feature  # pylint: disable=import-error

            edges = (
                feature.canny(
                    embedded,
                    sigma=4,
                    use_quantiles=False,
                    low_threshold=0.7,
                    high_threshold=1.5,
                )
                * 1
            )
            edges = np.where(edges[int(embed_size / 2)] == 1)[0]
            low_edges.append(np.min(edges))
            high_edges.append(np.max(edges))

        if indices:
            return np.array(low_edges), np.array(high_edges), rebinned.coords["eV"]

        delta = self._obj.G.stride(generic_dim_names=False)

        low_edges = (
            np.array(low_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0]
        )
        high_edges = (
            np.array(high_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0]
        )

        return low_edges, high_edges, rebinned.coords["eV"]

    def zero_spectrometer_edges(self, cut_margin=None, interp_range=None, low=None, high=None):
        if low is not None:
            assert high is not None
            assert len(low) == len(high) == 2

            low_edges = low
            high_edges = high

        (
            low_edges,
            high_edges,
            rebinned_eV_coord,
        ) = self.find_spectrum_angular_edges_full(indices=True)

        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        if cut_margin is None:
            if "pixel" in self._obj.dims:
                cut_margin = 50
            else:
                cut_margin = int(0.08 / self._obj.G.stride(generic_dim_names=False)[angular_dim])
        else:
            if isinstance(cut_margin, float):
                assert angular_dim == "phi"
                cut_margin = int(
                    cut_margin / self._obj.G.stride(generic_dim_names=False)[angular_dim]
                )

        if interp_range is not None:
            low_edge = xr.DataArray(low_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
            high_edge = xr.DataArray(high_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
            low_edge = low_edge.sel(eV=interp_range)
            high_edge = high_edge.sel(eV=interp_range)
            import pdb

            pdb.set_trace()

        other_dims = list(self._obj.dims)
        other_dims.remove("eV")
        other_dims.remove(angular_dim)
        copied = self._obj.copy(deep=True).transpose(*(["eV", angular_dim] + other_dims))

        low_edges += cut_margin
        high_edges -= cut_margin

        for i, energy in enumerate(copied.coords["eV"].values):
            index = np.searchsorted(rebinned_eV_coord, energy)
            other = index + 1
            if other >= len(rebinned_eV_coord):
                other = len(rebinned_eV_coord) - 1
                index = len(rebinned_eV_coord) - 2

            low = int(np.interp(energy, rebinned_eV_coord, low_edges))
            high = int(np.interp(energy, rebinned_eV_coord, high_edges))
            copied.values[i, 0:low] = 0
            copied.values[i, high:-1] = 0

        return copied

    def sum_other(self, dim_or_dims, keep_attrs=False):
        if isinstance(dim_or_dims, str):
            dim_or_dims = [dim_or_dims]

        return self._obj.sum(
            [d for d in self._obj.dims if d not in dim_or_dims], keep_attrs=keep_attrs
        )

    def mean_other(self, dim_or_dims, keep_attrs=False):
        if isinstance(dim_or_dims, str):
            dim_or_dims = [dim_or_dims]

        return self._obj.mean(
            [d for d in self._obj.dims if d not in dim_or_dims], keep_attrs=keep_attrs
        )

    def find_spectrum_angular_edges(self, indices=False):
        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        energy_edge = self.find_spectrum_energy_edges()
        energy_slice = slice(np.max(energy_edge) - 0.1, np.max(energy_edge))
        near_ef = self._obj.sel(eV=energy_slice).sum(
            [d for d in self._obj.dims if d not in [angular_dim]]
        )

        embed_size = 20
        embedded = np.ndarray(shape=[embed_size] + list(near_ef.values.shape))
        embedded[:] = near_ef.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        # try to avoid dependency conflict with numpy v0.16
        from skimage import feature  # pylint: disable=import-error

        edges = (
            feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True, low_threshold=0.2) * 1
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.G.stride(generic_dim_names=False)
        return edges * delta[angular_dim] + self._obj.coords[angular_dim].values[0]

    def trimmed_selector(self):
        raise NotImplementedError

    def wide_angle_selector(self, include_margin=True):
        edges = self.find_spectrum_angular_edges()
        low_edge, high_edge = np.min(edges), np.max(edges)

        # go and build in a small margin
        if include_margin:
            if "pixels" in self._obj.dims:
                low_edge += 50
                high_edge -= 50
            else:
                low_edge += 0.05
                high_edge -= 0.05

        return slice(low_edge, high_edge)

    def narrow_angle_selector(self):
        raise NotImplementedError

    def meso_effective_selector(self):
        energy_edge = self.find_spectrum_energy_edges()
        return slice(np.max(energy_edge) - 0.3, np.max(energy_edge) - 0.1)

    def region_sel(self, *regions):
        def process_region_selector(selector: Union[slice, DesignatedRegions], dimension_name: str):
            if isinstance(selector, slice):
                return selector

            # need to read out the region
            options = {
                "eV": (
                    DesignatedRegions.ABOVE_EF,
                    DesignatedRegions.BELOW_EF,
                    DesignatedRegions.EF_NARROW,
                    DesignatedRegions.MESO_EF,
                    DesignatedRegions.MESO_EFFECTIVE_EF,
                    DesignatedRegions.ABOVE_EFFECTIVE_EF,
                    DesignatedRegions.BELOW_EFFECTIVE_EF,
                    DesignatedRegions.EFFECTIVE_EF_NARROW,
                ),
                "phi": (
                    DesignatedRegions.NARROW_ANGLE,
                    DesignatedRegions.WIDE_ANGLE,
                    DesignatedRegions.TRIM_EMPTY,
                ),
            }

            options_for_dim = options.get(dimension_name, [d for d in DesignatedRegions])
            assert selector in options_for_dim

            # now we need to resolve out the region
            resolution_methods = {
                DesignatedRegions.ABOVE_EF: slice(0, None),
                DesignatedRegions.BELOW_EF: slice(None, 0),
                DesignatedRegions.EF_NARROW: slice(-0.1, 0.1),
                DesignatedRegions.MESO_EF: slice(-0.3, -0.1),
                DesignatedRegions.MESO_EFFECTIVE_EF: self.meso_effective_selector,
                # Implement me
                # DesignatedRegions.TRIM_EMPTY: ,
                DesignatedRegions.WIDE_ANGLE: self.wide_angle_selector,
                # DesignatedRegions.NARROW_ANGLE: self.narrow_angle_selector,
            }
            resolution_method = resolution_methods[selector]
            if isinstance(resolution_method, slice):
                return resolution_method
            if callable(resolution_method):
                return resolution_method()

            raise NotImplementedError("Unable to determine resolution method.")

        obj = self._obj

        def unpack_dim(dim_name):
            if dim_name == "angular":
                return "pixel" if "pixel" in obj.dims else "phi"

            return dim_name

        for region in regions:
            region = {unpack_dim(k): v for k, v in normalize_region(region).items()}

            # remove missing dimensions from selection for permissiveness
            # and to transparent composing of regions
            region = {k: process_region_selector(v, k) for k, v in region.items() if k in obj.dims}
            obj = obj.sel(**region)

        return obj

    def fat_sel(self, widths: Optional[Dict[str, Any]] = None, **kwargs) -> xr.DataArray:
        """Allows integrating a selection over a small region.

        The produced dataset will be normalized by dividing by the number
        of slices integrated over.

        This can be used to produce temporary datasets that have reduced
        uncorrelated noise.

        Args:
            widths: Override the widths for the slices. Resonable defaults are used otherwise. Defaults to None.
            kwargs: slice dict. Has the same function as xarray.DataArray.sel

        Returns:
            The data after selection.
        """
        if widths is None:
            widths = {}

        default_widths = {
            "eV": 0.05,
            "phi": 2,
            "beta": 2,
            "theta": 2,
            "kx": 0.02,
            "ky": 0.02,
            "kp": 0.02,
            "kz": 0.1,
        }

        extra_kwargs = {k: v for k, v in kwargs.items() if k not in self._obj.dims}
        slice_kwargs = {k: v for k, v in kwargs.items() if k not in extra_kwargs}
        slice_widths = {
            k: widths.get(k, extra_kwargs.get(k + "_width", default_widths.get(k)))
            for k in slice_kwargs
        }

        slices = {
            k: slice(v - slice_widths[k] / 2, v + slice_widths[k] / 2)
            for k, v in slice_kwargs.items()
        }

        sliced = self._obj.sel(**slices)
        thickness = np.product([len(sliced.coords[k]) for k in slice_kwargs.keys()])
        normalized = sliced.sum(slices.keys(), keep_attrs=True) / thickness
        for k, v in slices.items():
            normalized.coords[k] = (v.start + v.stop) / 2
        normalized.attrs.update(self._obj.attrs.copy())
        return normalized

    @property
    def reference_settings(self):
        settings = self.spectrometer_settings or {}

        settings.update(
            {
                "hv": self.hv,
            }
        )

        return settings

    @property
    def beamline_settings(self):
        find_keys = {
            "entrance_slit": {
                "entrance_slit",
            },
            "exit_slit": {
                "exit_slit",
            },
            "hv": {
                "hv",
                "photon_energy",
            },
            "grating": {},
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        if self.endstation == "BL403":
            settings["grating"] = "HEG"  # for now assume we always use the first order light

        return settings

    @property
    def spectrometer_settings(self):
        find_keys = {
            "lens_mode": {
                "lens_mode",
            },
            "pass_energy": {
                "pass_energy",
            },
            "scan_mode": {
                "scan_mode",
            },
            "scan_region": {
                "scan_region",
            },
            "slit": {
                "slit",
                "slit_plate",
            },
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        if isinstance(settings.get("slit"), (float, np.float32, np.float64)):
            settings["slit"] = int(round(settings["slit"]))

        return settings

    @property
    def sample_pos(self):
        x, y, z = None, None, None
        try:
            x = self._obj.attrs["x"]
        except KeyError:
            pass
        try:
            y = self._obj.attrs["y"]
        except KeyError:
            pass
        try:
            z = self._obj.attrs["z"]
        except KeyError:
            pass

        def do_float(w):
            return float(w) if w is not None else None

        return (do_float(x), do_float(y), do_float(z))

    @property
    def sample_angles(self):
        return (
            # manipulator
            self.lookup_coord("beta"),
            self.lookup_coord("theta"),
            self.lookup_coord("chi"),
            # analyzer
            self.lookup_coord("phi"),
            self.lookup_coord("psi"),
            self.lookup_coord("alpha"),
        )

    @property
    def full_coords(self):
        full_coords = {}

        full_coords.update(dict(zip(["x", "y", "z"], self.sample_pos)))
        full_coords.update(
            dict(zip(["beta", "theta", "chi", "phi", "psi", "alpha"], self.sample_angles))
        )
        full_coords.update(
            {
                "hv": self.hv,
            }
        )

        full_coords.update(self._obj.coords)
        return full_coords

    @property
    def sample_info(self):
        return unwrap_xarray_dict(
            {
                "id": self._obj.attrs.get("sample_id"),
                "name": self._obj.attrs.get("sample_name"),
                "source": self._obj.attrs.get("sample_source"),
                "reflectivity": self._obj.attrs.get("sample_reflectivity"),
            }
        )

    @property
    def scan_info(self):
        return unwrap_xarray_dict(
            {
                "time": self._obj.attrs.get("time"),
                "date": self._obj.attrs.get("date"),
                "type": self.scan_type,
                "spectrum_type": self.spectrum_type,
                "experimenter": self._obj.attrs.get("experimenter"),
                "sample": self._obj.attrs.get("sample_name"),
            }
        )

    @property
    def experiment_info(self):
        return unwrap_xarray_dict(
            {
                "temperature": self._obj.attrs.get("temperature"),
                "temperature_cryotip": self._obj.attrs.get("temperature_cryotip"),
                "pressure": self._obj.attrs.get("pressure"),
                "polarization": self.probe_polarization,
                "photon_flux": self._obj.attrs.get("photon_flux"),
                "photocurrent": self._obj.attrs.get("photocurrent"),
                "probe": self._obj.attrs.get("probe"),
                "probe_detail": self._obj.attrs.get("probe_detail"),
                "analyzer": self._obj.attrs.get("analyzer"),
                "analyzer_detail": self.analyzer_detail,
            }
        )

    @property
    def pump_info(self):
        return unwrap_xarray_dict(
            {
                "pump_wavelength": self._obj.attrs.get("pump_wavelength"),
                "pump_energy": self._obj.attrs.get("pump_energy"),
                "pump_fluence": self._obj.attrs.get("pump_fluence"),
                "pump_pulse_energy": self._obj.attrs.get("pump_pulse_energy"),
                "pump_spot_size": (
                    self._obj.attrs.get("pump_spot_size_x"),
                    self._obj.attrs.get("pump_spot_size_y"),
                ),
                "pump_profile": self._obj.attrs.get("pump_profile"),
                "pump_linewidth": self._obj.attrs.get("pump_linewidth"),
                "pump_temporal_width": self._obj.attrs.get("pump_temporal_width"),
                "pump_polarization": self.pump_polarization,
            }
        )

    @property
    def probe_info(self):
        return unwrap_xarray_dict(
            {
                "probe_wavelength": self._obj.attrs.get("probe_wavelength"),
                "probe_energy": self._obj.coords["hv"],
                "probe_fluence": self._obj.attrs.get("probe_fluence"),
                "probe_pulse_energy": self._obj.attrs.get("probe_pulse_energy"),
                "probe_spot_size": (
                    self._obj.attrs.get("probe_spot_size_x"),
                    self._obj.attrs.get("probe_spot_size_y"),
                ),
                "probe_profile": self._obj.attrs.get("probe_profile"),
                "probe_linewidth": self._obj.attrs.get("probe_linewidth"),
                "probe_temporal_width": self._obj.attrs.get("probe_temporal_width"),
                "probe_polarization": self.probe_polarization,
            }
        )

    @property
    def laser_info(self):
        return {
            **self.probe_info,
            **self.pump_info,
            "repetition_rate": self._obj.attrs.get("repetition_rate"),
        }

    @property
    def analyzer_info(self) -> Dict[str, Any]:
        """General information about the photoelectron analyzer used."""
        return unwrap_xarray_dict(
            {
                "lens_mode": self._obj.attrs.get("lens_mode"),
                "lens_mode_name": self._obj.attrs.get("lens_mode_name"),
                "acquisition_mode": self._obj.attrs.get("acquisition_mode"),
                "pass_energy": self._obj.attrs.get("pass_energy"),
                "slit_shape": self._obj.attrs.get("slit_shape"),
                "slit_width": self._obj.attrs.get("slit_width"),
                "slit_number": self._obj.attrs.get("slit_number"),
                "lens_table": self._obj.attrs.get("lens_table"),
                "analyzer_type": self._obj.attrs.get("analyzer_type"),
                "mcp_voltage": self._obj.attrs.get("mcp_voltage"),
            }
        )

    @property
    def daq_info(self) -> Dict[str, Any]:
        """General information about the acquisition settings for an ARPES experiment."""
        return unwrap_xarray_dict(
            {
                "daq_type": self._obj.attrs.get("daq_type"),
                "region": self._obj.attrs.get("daq_region"),
                "region_name": self._obj.attrs.get("daq_region_name"),
                "center_energy": self._obj.attrs.get("daq_center_energy"),
                "prebinning": self.prebinning,
                "trapezoidal_correction_strategy": self._obj.attrs.get(
                    "trapezoidal_correction_strategy"
                ),
                "dither_settings": self._obj.attrs.get("dither_settings"),
                "sweep_settings": self.sweep_settings,
                "frames_per_slice": self._obj.attrs.get("frames_per_slice"),
                "frame_duration": self._obj.attrs.get("frame_duration"),
            }
        )

    @property
    def beamline_info(self) -> Dict[str, Any]:
        """Information about the beamline or light source used for a measurement."""
        return unwrap_xarray_dict(
            {
                "hv": self._obj.coords["hv"],
                "linewidth": self._obj.attrs.get("probe_linewidth"),
                "photon_polarization": self.probe_polarization,
                "undulator_info": self.undulator_info,
                "repetition_rate": self._obj.attrs.get("repetition_rate"),
                "beam_current": self._obj.attrs.get("beam_current"),
                "entrance_slit": self._obj.attrs.get("entrance_slit"),
                "exit_slit": self._obj.attrs.get("exit_slit"),
                "monochromator_info": self.monochromator_info,
            }
        )

    @property
    def sweep_settings(self) -> Dict[str, Any]:
        """For datasets acquired with swept acquisition settings, provides those settings."""
        return {
            "high_energy": self._obj.attrs.get("sweep_high_energy"),
            "low_energy": self._obj.attrs.get("sweep_low_energy"),
            "n_sweeps": self._obj.attrs.get("n_sweeps"),
            "step": self._obj.attrs.get("sweep_step"),
        }

    @property
    def probe_polarization(self) -> Tuple[float, float]:
        """Provides the probe polarization of the UV/x-ray source."""
        return (
            self._obj.attrs.get("probe_polarization_theta"),
            self._obj.attrs.get("probe_polarization_alpha"),
        )

    @property
    def pump_polarization(self) -> Tuple[float, float]:
        """For Tr-ARPES experiments, provides the pump polarization."""
        return (
            self._obj.attrs.get("pump_polarization_theta"),
            self._obj.attrs.get("pump_polarization_alpha"),
        )

    @property
    def prebinning(self) -> Dict[str, Any]:
        """Information about the prebinning performed during scan acquisition."""
        prebinning = {}
        for d in self._obj.indexes:
            if "{}_prebinning".format(d) in self._obj.attrs:
                prebinning[d] = self._obj.attrs["{}_prebinning".format(d)]

        return prebinning

    @property
    def monochromator_info(self) -> Dict[str, Any]:
        """Details about the monochromator used on the UV/x-ray source."""
        return {
            "grating_lines_per_mm": self._obj.attrs.get("grating_lines_per_mm"),
        }

    @property
    def undulator_info(self) -> Dict[str, Any]:
        """Details about the undulator for data performed at an undulator source."""
        return {
            "gap": self._obj.attrs.get("undulator_gap"),
            "z": self._obj.attrs.get("undulator_z"),
            "harmonic": self._obj.attrs.get("undulator_harmonic"),
            "polarization": self._obj.attrs.get("undulator_polarization"),
            "type": self._obj.attrs.get("undulator_type"),
        }

    @property
    def analyzer_detail(self) -> Dict[str, Any]:
        """Details about the analyzer, its capabilities, and metadata."""
        return {
            "name": self._obj.attrs.get("analyzer_name"),
            "parallel_deflectors": self._obj.attrs.get("parallel_deflectors"),
            "perpendicular_deflectors": self._obj.attrs.get("perpendicular_deflectors"),
            "type": self._obj.attrs.get("analyzer_type"),
            "radius": self._obj.attrs.get("analyzer_radius"),
        }

    @property
    def temp(self) -> Union[float, xr.DataArray]:
        """The temperature at which an experiment was performed."""
        prefered_attrs = [
            "TA",
            "ta",
            "t_a",
            "T_A",
            "T_1",
            "t_1",
            "t1",
            "T1",
            "temp",
            "temp_sample",
            "temp_cryotip",
            "temperature_sensor_b",
            "temperature_sensor_a",
        ]
        for attr in prefered_attrs:
            if attr in self._obj.attrs:
                return float(self._obj.attrs[attr])

        raise AttributeError("Could not read temperature off any standard attr")

    @property
    def condensed_attrs(self) -> Dict[str, Any]:
        """An attributes shortlist.

        Since we enforce camelcase on attributes, this is a reasonable filter that catches
        the ones we don't use very often.
        """
        return {k: v for k, v in self._obj.attrs.items() if k[0].islower()}

    @property
    def referenced_scans(self) -> pd.DataFrame:
        if self.spectrum_type == "map":
            df = self._obj.attrs["df"]
            return df[(df.spectrum_type != "map") & (df.ref_id == self._obj.id)]
        else:
            assert self.spectrum_type in {"ucut", "spem"}
            df = self.df_until_type(
                spectrum_type=(
                    "ucut",
                    "spem",
                )
            )
            return df

    def generic_fermi_surface(self, fermi_energy):
        return self.fat_sel(eV=fermi_energy)

    @property
    def fermi_surface(self):
        return self.fat_sel(eV=0)

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @staticmethod
    def dict_to_html(d):
        return """
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(
            rows="".join(["<tr><td>{}</td><td>{}</td></tr>".format(k, v) for k, v in d.items()])
        )

    def _repr_html_full_coords(self, coords):
        def coordinate_dataarray_to_flat_rep(value):
            if not isinstance(value, xr.DataArray):
                return value

            return "<span>{min:.3g}<strong> to </strong>{max:.3g}<strong> by </strong>{delta:.3g}</span>".format(
                min=value.min().item(),
                max=value.max().item(),
                delta=value.values[1] - value.values[0],
            )

        return ARPESAccessorBase.dict_to_html(
            {k: coordinate_dataarray_to_flat_rep(v) for k, v in coords.items()}
        )

    def _repr_html_spectrometer_info(self):
        skip_keys = {
            "dof",
        }
        ordered_settings = OrderedDict(self.spectrometer_settings)
        ordered_settings.update({k: v for k, v in self.spectrometer.items() if k not in skip_keys})

        return ARPESAccessorBase.dict_to_html(ordered_settings)

    @staticmethod
    def _repr_html_experimental_conditions(conditions):
        transforms = {
            "polarization": lambda p: {
                "p": "Linear Horizontal",
                "s": "Linear Vertical",
                "rc": "Right Circular",
                "lc": "Left Circular",
                "s-p": "Linear Dichroism",
                "p-s": "Linear Dichroism",
                "rc-lc": "Circular Dichroism",
                "lc-rc": "Circular Dichroism",
            }.get(p, p),
            "hv": "{} eV".format,
            "temp": "{} Kelvin".format,
        }

        id = lambda x: x

        return ARPESAccessorBase.dict_to_html(
            {k: transforms.get(k, id)(v) for k, v in conditions.items() if v is not None}
        )

    def _repr_html_(self):
        skip_data_vars = {
            "time",
        }

        if isinstance(self._obj, xr.Dataset):
            to_plot = [k for k in self._obj.data_vars.keys() if k not in skip_data_vars]
            to_plot = [k for k in to_plot if 1 <= len(self._obj[k].dims) < 3]
            to_plot = to_plot[:5]

            if to_plot:
                _, ax = plt.subplots(
                    1,
                    len(to_plot),
                    figsize=(
                        len(to_plot) * 3,
                        3,
                    ),
                )
                if len(to_plot) == 1:
                    ax = [ax]

                for i, plot_var in enumerate(to_plot):
                    self._obj[plot_var].plot(ax=ax[i])
                    fancy_labels(ax[i])
                    ax[i].set_title(plot_var.replace("_", " "))

                remove_colorbars()

        else:
            if 1 <= len(self._obj.dims) < 3:
                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                self._obj.plot(ax=ax)
                fancy_labels(ax)
                ax.set_title("")

                remove_colorbars()

        wrapper_style = 'style="display: flex; flex-direction: row;"'

        try:
            name = self.df_index
        except:
            if "id" in self._obj.attrs:
                name = "ID: " + str(self._obj.attrs["id"])[:9] + "..."
            else:
                name = "No name"

        warning = ""

        if len(self._obj.attrs) < 10:
            warning = ':  <span style="color: red;">Few Attributes, Data Is Summed?</span>'

        return """
        <header><strong>{name}{warning}</strong></header>
        <div {wrapper_style}>
        <details open>
            <summary>Experimental Conditions</summary>
            {conditions}
        </details>
        <details open>
            <summary>Full Coordinates</summary>
            {coordinates}
        </details>
        <details open>
            <summary>Spectrometer</summary>
            {spectrometer_info}
        </details>
        </div>
        """.format(
            name=name,
            warning=warning,
            wrapper_style=wrapper_style,
            conditions=self._repr_html_experimental_conditions(self.experimental_conditions),
            coordinates=self._repr_html_full_coords(
                {k: v for k, v in self.full_coords.items() if v is not None}
            ),
            spectrometer_info=self._repr_html_spectrometer_info(),
        )


@xr.register_dataarray_accessor("S")
class ARPESDataArrayAccessor(ARPESAccessorBase):
    """Spectrum related accessor for `xr.DataArray`."""

    def plot(self, *args, rasterized=True, **kwargs):
        """Utility delegate to `xr.DataArray.plot` which rasterizes."""
        if len(self._obj.dims) == 2:
            kwargs["rasterized"] = rasterized

        with plt.rc_context(rc={"text.usetex": False}):
            self._obj.plot(*args, **kwargs)

    def show(self, detached=False, **kwargs):
        """Opens the Qt based image tool."""
        import arpes.plotting.qt_tool

        arpes.plotting.qt_tool.qt_tool(self._obj, detached=detached)

    def show_d2(self, **kwargs):
        """Opens the Bokeh based second derivative image tool."""
        from arpes.plotting.all import CurvatureTool

        curve_tool = CurvatureTool(**kwargs)
        return curve_tool.make_tool(self._obj)

    def show_band_tool(self, **kwargs):
        """Opens the Bokeh based band placement tool."""
        from arpes.plotting.all import BandTool

        band_tool = BandTool(**kwargs)
        return band_tool.make_tool(self._obj)

    def fs_plot(self, pattern="{}.png", **kwargs):
        """Provides a reference plot of the approximate Fermi surface."""
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_fs".format(self.label))
            kwargs["out"] = out
        return plotting.labeled_fermi_surface(self._obj, **kwargs)

    def fermi_edge_reference_plot(self, pattern="{}.png", **kwargs) -> plt.Axes:
        """Provides a reference plot for a Fermi edge reference."""
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_fermi_edge_reference".format(self.label))
            kwargs["out"] = out

        return plotting.fermi_edge.fermi_edge_reference(self._obj, **kwargs)

    def _referenced_scans_for_spatial_plot(self, use_id=True, pattern="{}.png", **kwargs):
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_reference_scan_fs".format(label))
            kwargs["out"] = out

        return plotting.reference_scan_spatial(self._obj, **kwargs)

    def _referenced_scans_for_map_plot(self, use_id=True, pattern="{}.png", **kwargs):
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_reference_scan_fs".format(label))
            kwargs["out"] = out

        return plotting.reference_scan_fermi_surface(self._obj, **kwargs)

    def _referenced_scans_for_hv_map_plot(self, use_id=True, pattern="{}.png", **kwargs):
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_hv_reference_scan".format(label))
            out = "{}_hv_reference_scan.png".format(label)
            kwargs["out"] = out

        return plotting.hv_reference_scan(self._obj, **kwargs)

    def _simple_spectrum_reference_plot(self, use_id=True, pattern="{}.png", **kwargs):
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_spectrum_reference".format(label))
            kwargs["out"] = out

        return plotting.fancy_dispersion(self._obj, **kwargs)

    def cut_nan_coords(self) -> xr.DataArray:
        """Selects data where coordinates are not `nan`.

        Returns:
            The subset of the data where coordinates are not `nan`.
        """
        slices = dict()
        for cname, cvalue in self._obj.coords.items():
            try:
                end_ind = np.where(np.isnan(cvalue.values))[0][0]
                end_ind = None if end_ind == -1 else end_ind
                slices[cname] = slice(None, end_ind)
            except IndexError:
                pass

        return self._obj.isel(**slices)

    def nan_to_num(self, x: Any = 0) -> xr.DataArray:
        """Provides an `xarray` version of `numpy.nan_to_num`.

        Args:
            x: The fill value

        Returns:
            A copy of the data with nans filled in.
        """
        data = self._obj.copy(deep=True)
        assert isinstance(data, xr.DataArray)
        data.values[np.isnan(data.values)] = x
        return data

    def reference_plot(self, **kwargs) -> plt.Axes:
        """Generates a reference plot for this piece of data according to its spectrum type.

        Raises:
            NotImplementedError: If there is no standard approach for plotting this data.

        Returns:
            The axes which were used for plotting.
        """
        if self.spectrum_type == "map":
            return self._referenced_scans_for_map_plot(**kwargs)
        elif self.spectrum_type == "hv_map":
            return self._referenced_scans_for_hv_map_plot(**kwargs)
        elif self.spectrum_type == "spectrum":
            return self._simple_spectrum_reference_plot(**kwargs)
        elif self.spectrum_type in {"ucut", "spem"}:
            return self._referenced_scans_for_spatial_plot(**kwargs)
        else:
            raise NotImplementedError


NORMALIZED_DIM_NAMES = ["x", "y", "z", "w"]


@xr.register_dataset_accessor("G")
@xr.register_dataarray_accessor("G")
class GenericAccessorTools:
    _obj = None

    def round_coordinates(self, coords, as_indices: bool = False):
        data = self._obj
        rounded = {
            k: v.item()
            for k, v in data.sel(**coords, method="nearest").coords.items()
            if k in coords
        }

        if as_indices:
            rounded = {k: data.coords[k].index(v) for k, v in rounded.items()}

        return rounded

    def argmax_coords(self):
        data = self._obj
        raveled_idx = data.argmax().item()
        flat_indices = np.unravel_index(raveled_idx, data.values.shape)
        max_coords = {d: data.coords[d][flat_indices[i]].item() for i, d in enumerate(data.dims)}
        return max_coords

    def apply_over(self, fn, copy=True, **selections):
        data = self._obj

        if copy:
            data = data.copy(deep=True)

        try:
            transformed = fn(data.sel(**selections))
        except:
            transformed = fn(data.sel(**selections).values)

        if isinstance(transformed, xr.DataArray):
            transformed = transformed.values

        data.loc[selections] = transformed
        return data

    def to_unit_range(self, percentile=None):
        if percentile is None:
            norm = self._obj - self._obj.min()
            return norm / norm.max()

        percentile = min(percentile, 100 - percentile)
        low, high = np.percentile(
            self._obj,
            (
                percentile,
                100 - percentile,
            ),
        )
        norm = self._obj - low
        return norm / (high - low)

    def extent(self, *args, dims=None) -> Tuple[float, float, float, float]:
        """Returns an "extent" array that can be used to draw with plt.imshow."""
        if dims is None:
            if not args:
                dims = self._obj.dims
            else:
                dims = args

        assert len(dims) == 2 and "You must supply exactly two dims to `.G.extent` not {}".format(
            dims
        )
        return [
            self._obj.coords[dims[0]][0].item(),
            self._obj.coords[dims[0]][-1].item(),
            self._obj.coords[dims[1]][0].item(),
            self._obj.coords[dims[1]][-1].item(),
        ]

    def drop_nan(self):
        assert len(self._obj.dims) == 1

        mask = np.logical_not(np.isnan(self._obj.values))
        return self._obj.isel(**dict([[self._obj.dims[0], mask]]))

    def shift_coords(self, dims, shift):
        if not isinstance(shift, np.ndarray):
            shift = np.ones((len(dims),)) * shift

        def transform(data):
            new_shift = shift
            for _ in range(len(dims)):
                new_shift = np.expand_dims(new_shift, 0)

            return data + new_shift

        return self.transform_coords(dims, transform)

    def scale_coords(self, dims, scale):
        if not isinstance(scale, np.ndarray):
            n_dims = len(dims)
            scale = np.identity(n_dims) * scale
        elif len(scale.shape) == 1:
            scale = np.diag(scale)

        return self.transform_coords(dims, scale)

    def transform_coords(
        self, dims: List[str], transform: Union[np.ndarray, Callable]
    ) -> xr.DataArray:
        """Transforms the given coordinate values according to an arbitrary function.

        The transformation should either be a function
        from a len(dims) x size of raveled coordinate array to len(dims) x size of raveled_coordinate
        array or a linear transformation as a matrix which is multiplied into such an array.

        Params:
            dims: A List of dimensions that should be transformed
            transform: The transformation to apply, can either be a function, or a matrix

        Returns:
            An identical valued array over new coordinates.
        """
        as_array = np.stack([self._obj.data_vars[d].values for d in dims], axis=-1)

        if isinstance(transform, np.ndarray):
            transformed = np.dot(as_array, transform)
        else:
            transformed = transform(as_array)

        copied = self._obj.copy(deep=True)

        for d, arr in zip(dims, np.split(transformed, transformed.shape[-1], axis=-1)):
            copied.data_vars[d].values = np.squeeze(arr, axis=-1)

        return copied

    def filter_vars(self, f):
        return xr.Dataset(
            data_vars={k: v for k, v in self._obj.data_vars.items() if f(v, k)},
            attrs=self._obj.attrs,
        )

    def coordinatize(self, as_coordinate_name: Optional[str] = None) -> xr.DataArray:
        """Copies data into a coordinate's data, with an optional renaming.

        If you think of an array as a function c => f(c) from coordinates to values at
        those coordinates, this function replaces f by the identity to give c => c

        Remarkably, `coordinatize` is a word.

        For the most part, this is only useful when converting coordinate values into
        k-space "forward".

        Args:
            as_coordinate_name: A new coordinate name for the only dimension. Defaults to None.

        Returns:
            An array which consists of the mapping c => c.
        """
        assert len(self._obj.dims) == 1

        d = self._obj.dims[0]
        if as_coordinate_name is None:
            as_coordinate_name = d

        o = self._obj.rename(dict([[d, as_coordinate_name]])).copy(deep=True)
        o.coords[as_coordinate_name] = o.values

        return o

    def ravel(self) -> Dict[str, xr.DataArray]:
        """Converts to a flat representation where the coordinate values are also present.

        Extremely valuable for plotting a dataset with coordinates, X, Y and values Z(X,Y)
        on a scatter plot in 3D.

        By default the data is listed under the key 'data'.

        Returns:
            A dictionary mapping between coordinate names and their coordinate arrays.
            Additionally, there is a key "data" which maps to the `.values` attribute of the array.
        """
        assert isinstance(self._obj, xr.DataArray)

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        raveled_coordinates = dict(zip(dims, [cs.ravel() for cs in np.meshgrid(*coords_as_list)]))
        assert "data" not in raveled_coordinates
        raveled_coordinates["data"] = self._obj.values.ravel()

        return raveled_coordinates

    def meshgrid(self, as_dataset=False):
        assert isinstance(self._obj, xr.DataArray)

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        meshed_coordinates = dict(zip(dims, [cs for cs in np.meshgrid(*coords_as_list)]))
        assert "data" not in meshed_coordinates
        meshed_coordinates["data"] = self._obj.values

        if as_dataset:
            # this could use a bit of cleaning up
            faked = ["x", "y", "z", "w"]
            meshed_coordinates = {
                k: (faked[: len(v.shape)], v) for k, v in meshed_coordinates.items() if k != "data"
            }

            return xr.Dataset(meshed_coordinates)

        return meshed_coordinates

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts a (1D) `xr.DataArray` into two plain ``ndarray``s of their coordinate and data.

        Useful for rapidly converting into a format than can be `plt.scatter`ed
        or similar.

        Example:
            We can use this to quickly scatter a 1D dataset where one axis is the coordinate value.

            >>> plt.scatter(*data.G.as_arrays(), marker='s')  # doctest: +SKIP

        Returns:
            A tuple of the coordinate array (first index) and the data array (second index)
        """
        assert len(self._obj.dims) == 1

        return (self._obj.coords[self._obj.dims[0]].values, self._obj.values)

    def clean_outliers(self, clip=0.5):
        low, high = np.percentile(self._obj.values, [clip, 100 - clip])
        copied = self._obj.copy(deep=True)
        copied.values[copied.values < low] = low
        copied.values[copied.values > high] = high
        return copied

    def as_movie(self, time_dim=None, pattern="{}.png", **kwargs):
        if time_dim is None:
            time_dim = self._obj.dims[-1]

        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format("{}_animation".format(self._obj.S.label))
            kwargs["out"] = out
        return plotting.plot_movie(self._obj, time_dim, **kwargs)

    def filter_coord(
        self, coordinate_name: str, sieve: Callable[[Any, xr.DataArray], bool]
    ) -> xr.DataArray:
        """Filters a dataset along a coordinate.

        Sieve should be a function which accepts a coordinate value and the slice
        of the data along that dimension.

        Internally, the predicate function `sieve` is applied to the coordinate and slice to generate
        a mask. The mask is used to select from the data after iteration.

        An improvement here would support filtering over several coordinates.

        Args:
            coordinate_name: The coordinate which should be filtered.
            sieve: A predicate to be applied to the coordinate and data at that coordinate.

        Returns:
            A subset of the data composed of the slices which make the `sieve` predicate `True`.
        """
        mask = np.array(
            [
                i
                for i, c in enumerate(self._obj.coords[coordinate_name])
                if sieve(c, self._obj.isel(**dict([[coordinate_name, i]])))
            ]
        )
        return self._obj.isel(**dict([[coordinate_name, mask]]))

    def iterate_axis(self, axis_name_or_axes):
        if isinstance(axis_name_or_axes, int):
            axis_name_or_axes = self._obj.dims[axis_name_or_axes]

        if isinstance(axis_name_or_axes, str):
            axis_name_or_axes = [axis_name_or_axes]

        coord_iterators = [self._obj.coords[d].values for d in axis_name_or_axes]
        for indices in itertools.product(*[range(len(c)) for c in coord_iterators]):
            cut_coords = [cs[index] for cs, index in zip(coord_iterators, indices)]
            coords_dict = dict(zip(axis_name_or_axes, cut_coords))
            yield coords_dict, self._obj.sel(method="nearest", **coords_dict)

    def map_axes(self, axes, fn, dtype=None, **kwargs):
        if isinstance(self._obj, xr.Dataset):
            raise TypeError(
                "map_axes can only work on xr.DataArrays for now because of "
                "how the type inference works"
            )
        obj = self._obj.copy(deep=True)

        if dtype is not None:
            obj.values = np.ndarray(shape=obj.values.shape, dtype=dtype)

        type_assigned = False
        for coord, value in self.iterate_axis(axes):
            new_value = fn(value, coord)

            if dtype is None:
                if not type_assigned:
                    obj.values = np.ndarray(shape=obj.values.shape, dtype=new_value.data.dtype)
                    type_assigned = True

                obj.loc[coord] = new_value.values
            else:
                obj.loc[coord] = new_value

        return obj

    def transform(
        self,
        axes: Union[str, List[str]],
        transform_fn: Callable,
        dtype: DTypeLike = None,
        *args,
        **kwargs,
    ):
        """Applies a vectorized operation across a subset of array axes.

        Transform has similar semantics to matrix multiplication, the dimensions of the
        output can grow or shrink depending on whether the transformation is size preserving,
        grows the data, shinks the data, or leaves in place.

        Examples:
            As an example, let us suppose we have a function which takes the mean and
            variance of the data:

                [dimension], coordinate_value -> [{'mean', 'variance'}]

            And a dataset with dimensions [X, Y]. Then calling transform
            maps to a dataset with the same dimension X but where Y has been replaced by
            the length 2 label {'mean', 'variance'}. The full dimensions in this case are
            ['X', {'mean', 'variance'}].

            >>> data.G.transform('X', f).dims  # doctest: +SKIP
            ["X", "mean", "variance"]

        Please note that the transformed axes always remain in the data because they
        are iterated over and cannot therefore be modified.

        The transform function `transform_fn` must accept the coordinate of the
        marginal at the currently iterated point.

         Args:
            axes: Dimension/axis or set of dimensions to iterate over
            transform_fn: Transformation function that takes a DataArray into a new DataArray
            dtype: An optional type hint for the transformed data. Defaults to None.
            args: args to pass into transform_fn
            kwargs: kwargs to pass into transform_fn

        Raises:
            TypeError: When the underying object is an `xr.Dataset` instead of an `xr.DataArray`.
            This is due to a constraint related to type inference with a single passed dtype.


        Returns:
            The data consisting of applying `transform_fn` across the specified axes.

        """
        if isinstance(self._obj, xr.Dataset):
            raise TypeError(
                "transform can only work on xr.DataArrays for now because of "
                "how the type inference works"
            )

        dest = None
        for coord, value in self.iterate_axis(axes):
            new_value = transform_fn(value, coord, *args, **kwargs)

            if dest is None:
                new_value = transform_fn(value, coord, *args, **kwargs)

                original_dims = [d for d in self._obj.dims if d not in value.dims]
                original_shape = [self._obj.shape[self._obj.dims.index(d)] for d in original_dims]
                original_coords = {k: v for k, v in self._obj.coords.items() if k not in value.dims}

                full_shape = original_shape + list(new_value.shape)

                new_coords = original_coords
                new_coords.update(
                    {k: v for k, v in new_value.coords.items() if k not in original_coords}
                )
                new_dims = original_dims + list(new_value.dims)
                dest = xr.DataArray(
                    np.zeros(full_shape, dtype=dtype or new_value.data.dtype),
                    coords=new_coords,
                    dims=new_dims,
                )

            dest.loc[coord] = new_value

        return dest

    def map(self, fn, **kwargs):
        return apply_dataarray(self._obj, np.vectorize(fn, **kwargs))

    def enumerate_iter_coords(self):
        coords_list = [self._obj.coords[d].values for d in self._obj.dims]
        for indices in itertools.product(*[range(len(c)) for c in coords_list]):
            cut_coords = [cs[index] for cs, index in zip(coords_list, indices)]
            yield indices, dict(zip(self._obj.dims, cut_coords))

    def iter_coords(self, dim_names=None):
        if dim_names is None:
            dim_names = self._obj.dims
        for ts in itertools.product(*[self._obj.coords[d].values for d in self._obj.dims]):
            yield dict(zip(self._obj.dims, ts))

    def range(self, generic_dim_names=True):
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_ranges = [(np.min(coord.values), np.max(coord.values)) for coord in indexed_coords]

        dim_names = self._obj.dims
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        return dict(zip(dim_names, indexed_ranges))

    def stride(self, *args, generic_dim_names=True):
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_strides = [coord.values[1] - coord.values[0] for coord in indexed_coords]

        dim_names = self._obj.dims
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        result = dict(zip(dim_names, indexed_strides))

        if args:
            if len(args) == 1:
                if not isinstance(args[0], str):
                    # if passed list of strs as argument
                    result = [result[selected_names] for selected_names in args[0]]
                else:
                    # if passed single name as argument
                    result = result[args[0]]
            else:
                # if passed several names as arguments
                result = [result[selected_names] for selected_names in args]

        return result

    def shift_by(self, other: xr.DataArray, shift_axis=None, zero_nans=True, shift_coords=False):
        # for now we only support shifting by a one dimensional array

        data = self._obj.copy(deep=True)

        by_axis = other.dims[0]
        assert len(other.dims) == 1
        assert len(other.coords[by_axis]) == len(data.coords[by_axis])

        if shift_coords:
            mean_shift = np.mean(other.values)
            other -= mean_shift

        if shift_axis is None:
            option_dims = list(data.dims)
            option_dims.remove(by_axis)
            assert len(option_dims) == 1
            shift_axis = option_dims[0]

        shift_amount = -other.values / data.G.stride(generic_dim_names=False)[shift_axis]

        shifted_data = arpes.utilities.math.shift_by(
            data.values,
            shift_amount,
            axis=list(data.dims).index(shift_axis),
            by_axis=list(data.dims).index(by_axis),
            order=1,
        )

        if zero_nans:
            shifted_data[np.isnan(shifted_data)] = 0

        built_data = xr.DataArray(
            shifted_data,
            data.coords,
            data.dims,
            attrs=data.attrs.copy(),
        )

        if shift_coords:
            built_data = built_data.assign_coords(
                **dict([[shift_axis, data.coords[shift_axis] + mean_shift]])
            )

        return built_data

    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj


@xr.register_dataarray_accessor("X")
class SelectionToolAccessor:
    _obj = None

    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj

    def max_in_window(self, around: xr.DataArray, window: Union[float, int], n_iters: int = 1):
        # TODO: refactor into a transform and finish the transform refactor to allow
        # simultaneous iteration
        destination = around.copy(deep=True) * 0

        # should only be one!
        (other_dim,) = list(set(self._obj.dims).difference(around.dims))

        for coord, value in around.G.iterate_axis(around.dims):
            value = value.item()
            marg = self._obj.sel(**coord)

            if isinstance(value, float):
                marg = marg.sel(**dict([[other_dim, slice(value - window, value + window)]]))
            else:
                marg = marg.isel(**dict([[other_dim, slice(value - window, value + window)]]))

            destination.loc[coord] = marg.coords[other_dim][marg.argmax().item()]

        if n_iters > 1:
            return self.max_in_window(destination, window, n_iters - 1)

        return destination

    def first_exceeding(
        self, dim, value: float, relative=False, reverse=False, as_index=False
    ) -> xr.DataArray:
        data = self._obj

        if relative:
            data = data / data.max(dim)

        cond = data > value
        reindex = data.coords[dim]

        if reverse:
            reindex = np.flip(reindex)
            cond = np.flip(cond, data.dims.index(dim)).values

        indices = cond.argmax(axis=data.dims.index(dim))
        if as_index:
            new_values = indices
            if reverse:
                new_values = -new_values + len(reindex) - 1
        else:
            new_values = reindex[indices]

        try:
            new_values = new_values.values
        except AttributeError:
            pass

        return data.isel(**dict([[dim, 0]])).S.with_values(new_values)

    def last_exceeding(self, dim, value: float, relative=False) -> xr.DataArray:
        return self.first_exceeding(dim, value, relative=relative, reverse=False)


@xr.register_dataset_accessor("F")
class ARPESDatasetFitToolAccessor:
    _obj = None

    def __init__(self, xarray_obj: DataType):
        self._obj = xarray_obj

    def eval(self, *args, **kwargs):
        return self._obj.results.G.map(lambda x: x.eval(*args, **kwargs))

    def show(self, detached=False):
        from arpes.plotting.fit_tool import fit_tool

        fit_tool(self._obj, detached=detached)

    @property
    def broadcast_dimensions(self) -> List[str]:
        """Returns the dimensions which were used in the fitting process.

        This is a sibling property to `fit_dimensions`.

        Returns:
            The list of the dimensions which were used in any individual fit.
            For example, a broadcast of MDCs across energy on a dataset with dimensions
            `["eV", "kp"]` would produce `["kp"]`.
        """
        return list(self._obj.results.dims)

    @property
    def fit_dimensions(self) -> List[str]:
        """Returns the dimensions which were broadcasted across, as opposed to fit across.

        This is a sibling property to `broadcast_dimensions`.

        Returns:
            The list of the dimensions which were **not** used in any individual fit.
            For example, a broadcast of MDCs across energy on a dataset with dimensions
            `["eV", "kp"]` would produce `["eV"]`.
        """
        return list(set(self._obj.data.dims).difference(self._obj.results.dims))

    def best_fits(self) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.best_fits`.

        Orders the fits into a raveled array by the MSE error.
        """
        return self._obj.results.F.best_fits()

    def worst_fits(self) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.worst_fits`.

        Orders the fits into a raveled array by the MSE error.
        """
        return self._obj.results.F.worst_fits()

    def mean_square_error(self) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.mean_square_error`.

        Calculates the mean square error of the fit across the fit
        axes for all model result instances in the collection.
        """
        return self._obj.results.F.mean_square_error()

    def p(self, param_name: str) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.p`.

        Collects the value of a parameter from curve fitting.

        Across an array of fits, walks parameters to collect the value
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the value found by the fitting routine.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        return self._obj.results.F.p(param_name)

    def s(self, param_name: str) -> xr.DataArray:
        """Alias for `ARPESFitToolsAccessor.s`.

        Collects the standard deviation of a parameter from fitting.

        Across an array of fits, walks parameters to collect the standard error
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the floating point value for the fits.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        return self._obj.results.F.s(param_name)

    def plot_param(self, param_name: str, **kwargs):
        """Alias for `ARPESFitToolsAccessor.plot_param`.

        Creates a scatter plot of a parameter from a multidimensional curve fit.

        Args:
            param_name: The name of the parameter which should be plotted
            kwargs: Passed to plotting routines to provide user control
        """
        return self._obj.results.F.plot_param(param_name, **kwargs)


@xr.register_dataarray_accessor("F")
class ARPESFitToolsAccessor:
    """Utilities related to examining curve fits."""

    _obj = None

    def __init__(self, xarray_obj: DataType):
        """Initialization hook for xarray.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        self._obj = xarray_obj

    def plot_param(self, param_name: str, **kwargs):
        """Creates a scatter plot of a parameter from a multidimensional curve fit.

        Args:
            param_name: The name of the parameter which should be plotted
            kwargs: Passed to plotting routines to provide user control
        """
        plot_parameter(self._obj, param_name, **kwargs)

    def param_as_dataset(self, param_name: str) -> xr.Dataset:
        """Maps from `lmfit.ModelResult` to a Dict parameter summary.

        Args:
            param_name: The parameter which should be summarized.

        Returns:
            A dataset consisting of two arrays: "value" and "error"
            which are the fit value and standard error on the parameter
            requested.
        """
        return xr.Dataset(
            {
                "value": self.p(param_name),
                "error": self.s(param_name),
            }
        )

    def show(self, detached: bool = False):
        """Opens a Bokeh based interactive fit inspection tool."""
        from arpes.plotting.fit_tool import fit_tool

        fit_tool(self._obj, detached=detached)

    def best_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error."""
        return self.order_stacked_fits(ascending=True)

    def worst_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error."""
        return self.order_stacked_fits(ascending=False)

    def mean_square_error(self) -> xr.DataArray:
        """Calculates the mean square error of the fit across fit axes.

        Producing a scalar metric of the error for all model result instances in
        the collection.
        """

        def safe_error(model_result_instance: Optional[lmfit.model.ModelResult]) -> float:
            if model_result_instance is None:
                return np.nan

            return (model_result_instance.residual ** 2).mean()

        return self._obj.G.map(safe_error)

    def order_stacked_fits(self, ascending=False) -> xr.DataArray:
        """Produces an ordered collection of `lmfit.ModelResult` instances.

        For multidimensional broadcasts, the broadcasted dimensions will be
        stacked for ordering to produce a 1D array of the results.

        Args:
            ascending: Whether the results should be ordered according to ascending
              mean squared error (best fits first) or descending error (worst fits first).

        Returns:
            An xr.DataArray instance with stacked axes whose values are the ordered models.
        """
        stacked = self._obj.stack({"by_error": self._obj.dims})
        error = stacked.F.mean_square_error()

        if not ascending:
            error = -error

        indices = np.argsort(error.values)
        return stacked[indices]

    def p(self, param_name: str) -> xr.DataArray:
        """Collects the value of a parameter from curve fitting.

        Across an array of fits, walks parameters to collect the value
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the value found by the fitting routine.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        return self._obj.G.map(param_getter(param_name), otypes=[np.float])

    def s(self, param_name: str) -> xr.DataArray:
        """Collects the standard deviation of a parameter from fitting.

        Across an array of fits, walks parameters to collect the standard error
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the floating point value for the fits.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        return self._obj.G.map(param_stderr_getter(param_name), otypes=[np.float])

    @property
    def bands(self) -> Dict[str, MultifitBand]:
        """Collects bands after a multiband fit.

        Returns:
            The collected bands.
        """
        band_names = self.band_names

        bands = {l: MultifitBand(label=l, data=self._obj) for l in band_names}

        return bands

    @property
    def band_names(self) -> Set[str]:
        """Collects the names of the bands from a multiband fit.

        Heuristically, a band is defined as a dispersive peak so we look for
        prefixes corresponding to parameter names which contain `"center"`.

        Returns:
            The collected prefix names for the bands.

            For instance, if the param name `"a_center"`, the return value
            would contain `"a_"`.
        """
        collected_band_names = set()

        for item in self._obj.values.ravel():
            if item is None:
                continue

            band_names = [k[:-6] for k in item.params.keys() if "center" in k]
            collected_band_names = collected_band_names.union(set(band_names))

        return collected_band_names

    @property
    def parameter_names(self) -> Set[str]:
        """Collects the parameter names for a multidimensional fit.

        Assumes that the model used is the same for all ``lmfit.ModelResult``s
        so that we can merely extract the parameter names from a single non-null
        result.

        Returns:
            A set of all the parameter names used in a curve fit.
        """
        collected_parameter_names = set()

        for item in self._obj.values.ravel():
            if item is None:
                continue

            param_names = [k for k in item.params.keys()]
            collected_parameter_names = collected_parameter_names.union(set(param_names))

        return collected_parameter_names


@xr.register_dataset_accessor("S")
class ARPESDatasetAccessor(ARPESAccessorBase):
    """Spectrum related accessor for `xr.Dataset`."""

    def __getattr__(self, item: str) -> Any:
        """Forward attribute access to the spectrum, if necessary.

        Args:
            item: Attribute name

        Returns:
            The attribute after lookup on the default spectrum
        """
        return getattr(self._obj.S.spectrum.S, item)

    def polarization_plot(self, **kwargs) -> plt.Axes:
        """Creates a spin polarization plot.

        Returns:
            The axes which were plotted onto for customization.
        """
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = "{}_spin_polarization.png".format(self.label)
            kwargs["out"] = out
        return plotting.spin_polarized_spectrum(self._obj, **kwargs)

    @property
    def is_spatial(self) -> bool:
        """Predicate indicating whether the dataset is a spatial scanning dataset.

        Returns:
            True if the dataset has dimensions indicating it is a spatial scan.
            False otherwise
        """
        try:
            return self.spectrum.S.is_spatial
        except Exception:
            # self.spectrum may be None, in which case it is not a spatial scan
            return False

    @property
    def spectrum(self) -> Optional[xr.DataArray]:
        """Isolates a single spectrum from a dataset.

        This is a convenience method which is typically used in startup for
        tools and analysis routines which need to operate on a single
        piece of data. As an example, the image browser `qt_tool` needs
        an `xr.DataArray` to operate but will accept an `xr.Dataset`
        which it will attempt to resolve to a single spectrum.

        In practice, we filter data variables by whether they contain "spectrum"
        in the name before selecting the one with the largest pixel volume.
        This is a heuristic which tries to guarantee we select ARPES data
        above XPS data, if they were collected together.

        Returns:
            A spectrum found in the dataset, if one can be isolated.

            In the case that several candidates are found, a single spectrum
            is selected among the candidates.

            Attributes from the parent dataset are assigned onto the selected
            array as a convenience.
        """
        spectrum = None
        if "spectrum" in self._obj.data_vars:
            spectrum = self._obj.spectrum
        elif "raw" in self._obj.data_vars:
            spectrum = self._obj.raw
        elif "__xarray_dataarray_variable__" in self._obj.data_vars:
            spectrum = self._obj.__xarray_dataarray_variable__
        else:
            candidates = self.spectra
            if candidates:
                spectrum = candidates[0]
                best_volume = np.prod(spectrum.shape)
                for c in candidates[1:]:
                    volume = np.prod(c.shape)
                    if volume > best_volume:
                        spectrum = c
                        best_volume = volume

        if spectrum is not None and "df" not in spectrum.attrs:
            spectrum.attrs["df"] = self._obj.attrs.get("df", None)

        return spectrum

    @property
    def spectra(self) -> List[xr.DataArray]:
        """Collects the variables which are likely spectra.

        Returns:
            The subset of the data_vars which have dimensions indicating
            that they are spectra.
        """
        spectra = []
        for dv in list(self._obj.data_vars):
            if "spectrum" in dv:
                spectra.append(self._obj[dv])

        return spectra

    @property
    def spectrum_type(self) -> str:
        """Gives a heuristic estimate of what kind of data is contained by the spectrum.

        Returns:
            The kind of data, coarsely
        """
        try:
            # this isn't the smartest thing in the world,
            # but it should allow some old code to keep working on datasets transparently
            return self.spectrum.S.spectrum_type
        except Exception:
            return "dataset"

    @property
    def degrees_of_freedom(self) -> Set[str]:
        """The collection of all degrees of freedom.

        Equivalently, dimensions on a piece of data.

        Returns:
            All degrees of freedom as a set.
        """
        return set(self.spectrum.dims)

    @property
    def spectrum_degrees_of_freedom(self) -> Set[str]:
        """Collects the spectrometer degrees of freedom.

        Spectrometer degrees of freedom are any which would be collected by an ARToF
        and their momentum equivalents.

        Returns:
            The collection of spectrum degrees of freedom.
        """
        return self.degrees_of_freedom.intersection({"eV", "phi", "pixel", "kx", "kp", "ky"})

    @property
    def scan_degrees_of_freedom(self) -> Set[str]:
        """Collects the scan degrees of freedom.

        Scan degrees of freedom are all of the degrees of freedom which are not recorded
        by the spectrometer but are "scanned over". This includes spatial axes,
        temperature, etc.

        Returns:
            The collection of scan degrees of freedom represented in the array.
        """
        return self.degrees_of_freedom.difference(self.spectrum_degrees_of_freedom)

    def reference_plot(self, **kwargs):
        """Creates reference plots for a dataset.

        A bit of a misnomer because this actually makes many plots. For full datasets,
        the relevant components are:

        #. Temperature as function of scan DOF
        #. Photocurrent as a function of scan DOF
        #. Photocurrent normalized + unnormalized figures, in particular
            #. The reference plots for the photocurrent normalized spectrum
            #. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF
              integrated over E, phi
            #. For delay scans:
                #. Fermi location as a function of scan DoF, integrated over phi
                #. Subtraction scans
        #. For spatial scans:
            #. energy/angle integrated spatial maps with subsequent measurements indicated
            #. energy/angle integrated FS spatial maps with subsequent measurements indicated

        Args:
            kwargs: Passed to plotting routines to provide user control
        """
        scan_dofs_integrated = self._obj.sum(*list(self.scan_degrees_of_freedom))
        original_out = kwargs.get("out")

        # make figures for temperature, photocurrent, delay
        make_figures_for = ["T", "IG_nA", "current", "photocurrent"]
        name_normalization = {
            "T": "T",
            "IG_nA": "photocurrent",
            "current": "photocurrent",
        }

        for figure_item in make_figures_for:
            if figure_item not in self._obj.data_vars:
                continue

            name = name_normalization.get(figure_item, figure_item)
            data_var = self._obj[figure_item]
            out = "{}_{}_spec_integrated_reference.png".format(self.label, name)
            plotting.scan_var_reference_plot(data_var, title="Reference {}".format(name), out=out)

        # may also want to make reference figures summing over cycle, or summing over beta

        # make photocurrent normalized figures
        try:
            normalized = self._obj / self._obj.IG_nA
            normalized.S.make_spectrum_reference_plots(prefix="norm_PC_", out=True)
        except:
            pass

        self.make_spectrum_reference_plots(out=True)

    def make_spectrum_reference_plots(self, prefix: str = "", **kwargs):
        """Creates photocurrent normalized + unnormalized figures.

        Creates:
        #. The reference plots for the photocurrent normalized spectrum
        #. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF integrated over E, phi
        #. For delay scans:

            #. Fermi location as a function of scan DoF, integrated over phi
            #. Subtraction scans

        Args:
            prefix: A prefix inserted into filenames to make them unique.
            kwargs: Passed to plotting routines to provide user control over plotting
                    behavior
        """
        self.spectrum.S.reference_plot(pattern=prefix + "{}.png", **kwargs)

        if self.is_spatial:
            referenced = self.referenced_scans

        if "cycle" in self._obj.coords:
            integrated_over_scan = self._obj.sum(*list(self.spectrum_degrees_of_freedom))
            integrated_over_scan.S.spectrum.S.reference_plot(
                pattern=prefix + "sum_spec_DoF_{}.png", **kwargs
            )

        if "delay" in self._obj.coords:
            dims = self.spectrum_degrees_of_freedom
            dims.remove("eV")
            angle_integrated = self._obj.sum(*list(dims))

            # subtraction scan
            self.spectrum.S.subtraction_reference_plots(pattern=prefix + "{}.png", **kwargs)
            angle_integrated.S.fermi_edge_reference_plots(pattern=prefix + "{}.png", **kwargs)

    def __init__(self, xarray_obj: xr.Dataset):
        """Initialization hook for xarray.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        super().__init__(xarray_obj)
        self._spectrum = None
