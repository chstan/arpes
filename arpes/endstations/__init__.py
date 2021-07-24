"""Plugin facility to read and normalize information from different sources to a common format."""
import warnings
import re

import numpy as np
import h5py
import xarray as xr
from astropy.io import fits

from pathlib import Path
from typing import Any, Dict, List, Union
import copy
import arpes.config
import arpes.constants
import os.path

from arpes.load_pxt import read_single_pxt, find_ses_files_associated
from arpes.utilities.dict import case_insensitive_get, rename_dataarray_attrs
from arpes.provenance import provenance_from_file
from arpes.endstations.fits_utils import find_clean_coords
from arpes.endstations.igor_utils import shim_wave_note
from arpes.repair import negate_energy


__all__ = [
    "endstation_name_from_alias",
    "endstation_from_alias",
    "add_endstation",
    "load_scan",
    "EndstationBase",
    "FITSEndstation",
    "HemisphericalEndstation",
    "SynchrotronEndstation",
    "SingleFileEndstation",
    "resolve_endstation",
]

_ENDSTATION_ALIASES = {}


class EndstationBase:
    """Implements the core features of ARPES data loading.

    A thorough documentation
    is available at `the plugin documentation <https://arpes.readthedocs.io/writing-plugins>`_.

    To summarize, a plugin has a few core jobs:

    1. Load data, including collating any data that is in a multi-file format
       This is accomplished with `.load`, which delegates loading `frames` (single files)
       to `load_single_frame`. Frame collation is then performed by `concatenate_frames`.
    2. Loading and attaching metadata.
    3. Normalizing metadata to standardized names. These are documented at the
       `data model documentation <https://arpes.readthedocs.io/spectra>`_.
    4. Ensuring all angles and necessary coordinates are attached to the data.
       Data should permit immediate conversion to angle space after being loaded.

    Plugins are in one-to-many correspondance with the values of the "location" column in
    analysis spreadsheets. This binding is provided by PRINCIPAL_NAME and ALIASES.

    The simplest way to normalize metadata is by renaming keys, but sometimes additional
    work is required. RENAME_KEYS is provided to make this simpler, and is implemented in
    scan post-processessing.
    """

    ALIASES = []
    PRINCIPAL_NAME = None
    ATTR_TRANSFORMS = {}
    MERGE_ATTRS = {}

    _SEARCH_DIRECTORIES = (
        "",
        "hdf5",
        "fits",
        "../Data",
        "../Data/hdf5",
        "../Data/fits",
    )
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )
    _TOLERATED_EXTENSIONS = {
        ".h5",
        ".nc",
        ".fits",
        ".pxt",
        ".nxs",
        ".txt",
    }
    _USE_REGEX = True

    # adjust as needed
    ENSURE_COORDS_EXIST = ["x", "y", "z", "theta", "beta", "chi", "hv", "alpha", "psi"]
    CONCAT_COORDS = ["hv", "chi", "psi", "timed_power", "tilt", "beta", "theta"]

    # phi because this happens sometimes at BL4 with core level scans
    SUMMABLE_NULL_DIMS = ["phi", "cycle"]

    RENAME_KEYS = {}

    @classmethod
    def is_file_accepted(cls, file, scan_desc) -> bool:
        """Determines whether this loader can load this file."""
        if os.path.exists(str(file)) and len(str(file).split(os.path.sep)) > 1:
            # looks like an actual file, we are going to just check that the extension is kosher
            # and that the filename matches something reasonable.
            p = Path(str(file))

            if p.suffix not in cls._TOLERATED_EXTENSIONS:
                return False

            for pattern in cls._SEARCH_PATTERNS:
                regex = re.compile(pattern.format(r"[0-9]+"))
                if regex.match(p.stem):
                    return True

            return False
        try:
            _ = cls.find_first_file(file, scan_desc)
            return True
        except ValueError:
            return False

    @classmethod
    def files_for_search(cls, directory) -> List[str]:
        """Filters files in a directory for candidate scans.

        Here, this just means collecting the ones with extensions acceptable to the loader.
        """
        return [
            f for f in os.listdir(directory) if os.path.splitext(f)[1] in cls._TOLERATED_EXTENSIONS
        ]

    @classmethod
    def find_first_file(cls, file, scan_desc, allow_soft_match=False):
        """Attempts to find a file associated to the scan given the user provided path or scan number.

        This is mostly done by regex matching over available options.
        Endstations which do not require further control here can just provide class attributes:

        * `._SEARCH_DIRECTORIES`: Defining which paths should be checked for scans
        * `._SEARCH_PATTERNS`: Defining acceptable filenames
        * `._USE_REGEX`: Controlling literal or regex filename checking
        * `._TOLERATED_EXTENSIONS`: Controlling whether files should be rejected based on their extension.
        """
        workspace = arpes.config.CONFIG["WORKSPACE"]
        workspace_path = os.path.join(workspace["path"], "data")
        workspace = workspace["name"]

        base_dir = workspace_path or os.path.join(arpes.config.DATA_PATH, workspace)
        dir_options = [os.path.join(base_dir, option) for option in cls._SEARCH_DIRECTORIES]

        # another plugin related option here is we can restrict the number of regexes by allowing plugins
        # to install regexes for particular endstations, if this is needed in the future it might be a good way
        # of preventing clashes where there is ambiguity in file naming scheme across endstations

        patterns = [re.compile(m.format(file)) for m in cls._SEARCH_PATTERNS]

        for dir in dir_options:
            try:
                files = cls.files_for_search(dir)

                if cls._USE_REGEX:
                    for p in patterns:
                        for f in files:
                            m = p.match(os.path.splitext(f)[0])
                            if m is not None:
                                if m.string == os.path.splitext(f)[0]:
                                    return os.path.join(dir, f)
                else:
                    for f in files:
                        if os.path.splitext(file)[0] == os.path.splitext(f)[0]:
                            return os.path.join(dir, f)
                        if allow_soft_match:
                            matcher = os.path.splitext(f)[0].split("_")[-1]
                            try:
                                if int(matcher) == int(file):
                                    return os.path.join(dir, f)  # soft match
                            except ValueError:
                                pass
            except FileNotFoundError:
                pass

        if str(file) and str(file)[0] == "f":  # try trimming the f off
            return cls.find_first_file(str(file)[1:], scan_desc, allow_soft_match=allow_soft_match)

        raise ValueError("Could not find file associated to {}".format(file))

    def concatenate_frames(self, frames=List[xr.Dataset], scan_desc: dict = None):
        """Performs concatenation of frames in multi-frame scans.

        The way this happens is that we look for an axis on which the frames are changing uniformly
        among a set of candidates (`.CONCAT_COORDS`). Then we delegate to xarray to perform the concatenation
        and clean up the merged coordinate.
        """
        if not frames:
            raise ValueError("Could not read any frames.")

        if len(frames) == 1:
            return frames[0]

        # determine which axis to stitch them together along, and then do this
        scan_coord = None
        max_different_values = -np.inf
        for possible_scan_coord in self.CONCAT_COORDS:
            coordinates = [f.attrs.get(possible_scan_coord, None) for f in frames]
            n_different_values = len(set(coordinates))
            if n_different_values > max_different_values and None not in coordinates:
                max_different_values = n_different_values
                scan_coord = possible_scan_coord

        assert scan_coord is not None

        for f in frames:
            f.coords[scan_coord] = f.attrs[scan_coord]

        frames.sort(key=lambda x: x.coords[scan_coord])
        return xr.concat(frames, scan_coord)

    def resolve_frame_locations(self, scan_desc: dict = None) -> List[str]:
        """Determine all files and frames associated to this piece of data.

        This always needs to be overridden in subclasses to handle data appropriately.
        """
        raise NotImplementedError(
            "You need to define resolve_frame_locations or subclass SingleFileEndstation."
        )

    def load_single_frame(
        self, frame_path: str = None, scan_desc: dict = None, **kwargs
    ) -> xr.Dataset:
        """Hook for loading a single frame of data.

        This always needs to be overridden in subclasses to handle data appropriately.
        """
        print(frame_path)
        return xr.Dataset()

    def postprocess(self, frame: xr.Dataset):
        """Performs frame level normalization of scan data.

        Here, we currently:
        1. Remove dimensions if they only have a single point, i.e. if the scan has shape [1,N] it
          gets squeezed to have size [N]
        2. Rename attributes
        """
        from arpes.utilities import rename_keys

        frame = xr.Dataset(
            {k: rename_dataarray_attrs(v, self.RENAME_KEYS) for k, v in frame.data_vars.items()},
            attrs=rename_keys(frame.attrs, self.RENAME_KEYS),
        )

        sum_dims = []
        for dim in frame.dims:
            if len(frame.coords[dim]) == 1 and dim in self.SUMMABLE_NULL_DIMS:
                sum_dims.append(dim)

        if sum_dims:
            frame = frame.sum(sum_dims, keep_attrs=True)

        return frame

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        """Perform final normalization of scan data.

        This defines the common codepaths for attaching extra information to scans at load time.
        Currently this means we:

        1. Attach a normalized "type" or "kind" of the spectrum indicating what sort of scan it is
        2. Ensure standard coordinates are represented
        3. Apply attribute renaming and attribute transformations defined by class attrs
        4. Ensure the scan endianness matches the system for performance reasons down the line
        """
        # attach the 'spectrum_type'
        # TODO move this logic into xarray extensions and customize here
        # only as necessary
        coord_names = tuple(sorted([c for c in data.dims if c != "cycle"]))

        spectrum_type = None
        if any(d in coord_names for d in {"x", "y", "z"}):
            coord_names = tuple(c for c in coord_names if c not in {"x", "y", "z"})
            spectrum_types = {
                ("eV",): "spem",
                ("eV", "phi"): "ucut",
            }
            spectrum_type = spectrum_types.get(coord_names)
        else:
            spectrum_types = {
                ("eV",): "xps",
                ("eV", "phi", "theta"): "map",
                ("eV", "phi", "psi"): "map",
                ("beta", "eV", "phi"): "map",
                ("eV", "hv", "phi"): "hv_map",
                ("eV", "phi"): "cut",
            }
            spectrum_type = spectrum_types.get(coord_names)

        if "phi" not in data.coords:
            # XPS
            data.coords["phi"] = 0
            for s in data.S.spectra:
                s.coords["phi"] = 0

        if spectrum_type is not None:
            data.attrs["spectrum_type"] = spectrum_type
            if "spectrum" in data.data_vars:
                data.spectrum.attrs["spectrum_type"] = spectrum_type

        ls = [data] + data.S.spectra
        for l in ls:
            for k, key_fn in self.ATTR_TRANSFORMS.items():
                if k in l.attrs:
                    transformed = key_fn(l.attrs[k])
                    if isinstance(transformed, dict):
                        l.attrs.update(transformed)
                    else:
                        l.attrs[k] = transformed

        for l in ls:
            for k, v in self.MERGE_ATTRS.items():
                if k not in l.attrs:
                    l.attrs[k] = v

        for l in ls:
            for c in self.ENSURE_COORDS_EXIST:
                if c not in l.coords:
                    if c in l.attrs:
                        l.coords[c] = l.attrs[c]
                    else:
                        warnings.warn(
                            f"Could not assign coordinate {c} from attributes, assigning np.nan instead."
                        )
                        l.coords[c] = np.nan

        for l in ls:
            if "chi" in l.coords and "chi_offset" not in l.attrs:
                l.attrs["chi_offset"] = l.coords["chi"].item()

        # go and change endianness and datatypes to something reasonable
        # this is done for performance reasons in momentum space conversion, primarily
        for k, v in data.data_vars.items():
            if not v.dtype.isnative:
                v.values = v.values.byteswap().newbyteorder()

        return data

    def load_from_path(self, path: Union[str, Path]) -> xr.Dataset:
        """Convenience wrapper around `.load` which references an explicit path."""
        path = str(path)
        return self.load(
            {
                "file": path,
                "location": self.PRINCIPAL_NAME,
            }
        )

    def load(self, scan_desc: dict = None, **kwargs) -> xr.Dataset:
        """Loads a scan from a single file or a sequence of files.

        This defines the contract and structure for standard data loading plugins:
        1. Search for files (`.resolve_frame_locations`)
        2. Load them sequentially (`.load_single_frame`)
        3. Apply cleaning code to each frame (`.postprocess`)
        4. Concatenate these loaded files  (`.concatenate_frames`)
        5. Apply postprocessing code to the concatenated dataset

        You can read more about the plugin system in the detailed documentation,
        but for the most part loaders just specializing one or more of these different steps
        as appropriate for a beamline.
        """
        resolved_frame_locations = self.resolve_frame_locations(scan_desc)
        resolved_frame_locations = [
            f if isinstance(f, str) else str(f) for f in resolved_frame_locations
        ]

        frames = [
            self.load_single_frame(fpath, scan_desc, **kwargs) for fpath in resolved_frame_locations
        ]
        frames = [self.postprocess(f) for f in frames]
        concatted = self.concatenate_frames(frames, scan_desc)
        concatted = self.postprocess_final(concatted, scan_desc)

        if "id" in scan_desc:
            concatted.attrs["id"] = scan_desc["id"]

        return concatted


class SingleFileEndstation(EndstationBase):
    """Abstract endstation which loads data from a single file.

    This just specializes the routine used to determine the location of files on disk.

    Unlike general endstations, if your data comes in a single file you can trust that the
    file given to you in the spreadsheet or direct load calls is all there is.
    """

    def resolve_frame_locations(self, scan_desc: dict = None):
        """Single file endstations just use the referenced file from the scan description."""
        if scan_desc is None:
            raise ValueError(
                "Must pass dictionary as file scan_desc to all endstation loading code."
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        p = Path(original_data_loc)
        return [p]


class SESEndstation(EndstationBase):
    """Provides collation and loading for Scienta's SESWrapper and endstations using it.

    These files have special frame names, at least at the beamlines Conrad has encountered.
    """

    def resolve_frame_locations(self, scan_desc: dict = None):
        if scan_desc is None:
            raise ValueError(
                "Must pass dictionary as file scan_desc to all endstation loading code."
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        p = Path(original_data_loc)
        return find_ses_files_associated(p)

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        name, ext = os.path.splitext(frame_path)

        if "nc" in ext:
            # was converted to hdf5/NetCDF format with Conrad's Igor scripts
            scan_desc = copy.deepcopy(scan_desc)
            scan_desc["path"] = frame_path
            return self.load_SES_nc(scan_desc=scan_desc, **kwargs)

        # it's given by SES PXT files
        pxt_data = negate_energy(read_single_pxt(frame_path))
        return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)

    def postprocess(self, frame: xr.Dataset):
        import arpes.xarray_extensions  # pylint: disable=unused-import, redefined-outer-name

        frame = super().postprocess(frame)
        return frame.assign_attrs(frame.S.spectrum.attrs)

    def load_SES_nc(self, scan_desc: dict = None, robust_dimension_labels=False, **kwargs):
        """Imports an hdf5 dataset exported from Igor that was originally generated in SESb format.

        In order to understand the structure of these files have a look at Conrad's saveSESDataset in
        Igor Pro.

        Args:
            scan_desc: Dictionary with extra information to attach to the xr.Dataset, must contain the location
              of the file
            robust_dimension_labels: safety control, used to load despite possibly malformed dimension names

        Returns:
            Loaded data.
        """
        scan_desc = copy.deepcopy(scan_desc)

        data_loc = scan_desc.get("path", scan_desc.get("file"))
        p = Path(data_loc)
        if not p.exists():
            data_loc = os.path.join(arpes.config.DATA_PATH, data_loc)

        wave_note = shim_wave_note(data_loc)
        f = h5py.File(data_loc, "r")

        primary_dataset_name = list(f)[0]
        # This is bugged for the moment in h5py due to an inability to read fixed length unicode strings
        # wave_note = f['/' + primary_dataset_name].attrs['IGORWaveNote']

        # Use dimension labels instead of
        dimension_labels = list(f["/" + primary_dataset_name].attrs["IGORWaveDimensionLabels"][0])
        if any(x == "" for x in dimension_labels):
            print(dimension_labels)

            if not robust_dimension_labels:
                raise ValueError(
                    "Missing dimension labels. Use robust_dimension_labels=True to override"
                )
            else:
                used_blanks = 0
                for i in range(len(dimension_labels)):
                    if dimension_labels[i] == "":
                        dimension_labels[i] = "missing{}".format(used_blanks)
                        used_blanks += 1

                print(dimension_labels)

        scaling = f["/" + primary_dataset_name].attrs["IGORWaveScaling"][-len(dimension_labels) :]
        raw_data = f["/" + primary_dataset_name][:]

        scaling = [
            np.linspace(scale[1], scale[1] + scale[0] * raw_data.shape[i], raw_data.shape[i])
            for i, scale in enumerate(scaling)
        ]

        dataset_contents = {}
        attrs = scan_desc.pop("note", {})
        attrs.update(wave_note)

        built_coords = dict(zip(dimension_labels, scaling))

        deg_to_rad_coords = {"theta", "beta", "phi", "alpha", "psi"}

        # the hemisphere axis is handled below
        built_coords = {
            k: c * (np.pi / 180) if k in deg_to_rad_coords else c for k, c in built_coords.items()
        }

        deg_to_rad_attrs = {"theta", "beta", "alpha", "psi", "chi"}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in attrs:
                attrs[angle_attr] = float(attrs[angle_attr]) * np.pi / 180

        dataset_contents["spectrum"] = xr.DataArray(
            raw_data,
            coords=built_coords,
            dims=dimension_labels,
            attrs=attrs,
        )

        provenance_from_file(
            dataset_contents["spectrum"],
            data_loc,
            {"what": "Loaded SES dataset from HDF5.", "by": "load_SES"},
        )

        return xr.Dataset(
            dataset_contents,
            attrs={**scan_desc, "name": primary_dataset_name},
        )


class FITSEndstation(EndstationBase):
    """Loads data from the .fits format produced by the MAESTRO software and derivatives.

    This ends up being somewhat complicated, because the FITS export is written in LabView and
    does not conform to the standard specification for the FITS archive format.

    Many of the intricacies here are in fact those shared between MAESTRO's format
    and the Lanzara Lab's format. Conrad does not foresee this as an issue, because it is
    unlikely that many other ARPES labs will adopt this data format moving forward, in
    light of better options derivative of HDF like the NeXuS format.
    """

    PREPPED_COLUMN_NAMES = {
        "time": "time",
        "Delay": "delay-var",  # these are named thus to avoid conflicts with the
        "Sample-X": "cycle-var",  # underlying coordinates
        "Mira": "pump_power",
        # insert more as needed
    }

    SKIP_COLUMN_NAMES = {
        "Phi",
        "null",
        "X",
        "Y",
        "Z",
        "mono_eV",
        "Slit Defl",
        "Optics Stage",
        "Scan X",
        "Scan Y",
        "Scan Z",
        # insert more as needed
    }

    SKIP_COLUMN_FORMULAS = {
        lambda name: True if ("beamview" in name or "IMAQdx" in name) else False,
    }

    RENAME_KEYS = {
        "Phi": "chi",
        "Beta": "beta",
        "Azimuth": "chi",
        "Pump_energy_uJcm2": "pump_fluence",
        "T0_ps": "t0_nominal",
        "W_func": "workfunction",
        "Slit": "slit",
        "LMOTOR0": "x",
        "LMOTOR1": "y",
        "LMOTOR2": "z",
        "LMOTOR3": "theta",
        "LMOTOR4": "beta",
        "LMOTOR5": "chi",
        "LMOTOR6": "alpha",
    }

    def resolve_frame_locations(self, scan_desc: dict = None):
        """These are stored as single files, so just use the one from the description."""
        if scan_desc is None:
            raise ValueError(
                "Must pass dictionary as file scan_desc to all endstation loading code."
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        return [original_data_loc]

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """Loads a scan from a single .fits file.

        This assumes the DAQ storage convention set by E. Rotenberg (possibly earlier authors)
        for the storage of ARPES data in FITS tables.

        This involves several complications:

        1. Hydrating/extracting coordinates from start/delta/n formats
        2. Extracting multiple scan regions
        3. Gracefully handling missing values
        4. Unwinding different scan conventions to common formats
        5. Handling early scan termination
        """
        # Use dimension labels instead of
        hdulist = fits.open(frame_path, ignore_missing_end=True)
        primary_dataset_name = None

        # Clean the header because sometimes out LabView produces improper FITS files
        for i in range(len(hdulist)):
            # This looks a little stupid, but because of confusing astropy internals actually works
            hdulist[i].header["UN_0_0"] = ""  # TODO This card is broken, this is not a good fix
            del hdulist[i].header["UN_0_0"]
            hdulist[i].header["UN_0_0"] = ""
            if "TTYPE2" in hdulist[i].header and hdulist[i].header["TTYPE2"] == "Delay":
                hdulist[i].header["TUNIT2"] = ""
                del hdulist[i].header["TUNIT2"]
                hdulist[i].header["TUNIT2"] = "ps"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdulist[i].verify("fix+warn")
                hdulist[i].header.update()
            # This actually requires substantially more work because it is lossy to information
            # on the unit that was encoded

        hdu = hdulist[1]

        scan_desc = copy.deepcopy(scan_desc)
        attrs = scan_desc.pop("note", scan_desc)
        attrs.update(dict(hdulist[0].header))

        drop_attrs = ["COMMENT", "HISTORY", "EXTEND", "SIMPLE", "SCANPAR", "SFKE_0"]
        for dropped_attr in drop_attrs:
            if dropped_attr in attrs:
                del attrs[dropped_attr]

        from arpes.utilities import rename_keys

        built_coords, dimensions, real_spectrum_shape = find_clean_coords(hdu, attrs, mode="MC")
        attrs = rename_keys(attrs, self.RENAME_KEYS)
        scan_desc = rename_keys(scan_desc, self.RENAME_KEYS)

        def clean_key_name(k: str) -> str:
            if "#" in k:
                k = k.replace("#", "num")

            return k

        attrs = {clean_key_name(k): v for k, v in attrs.items()}
        scan_desc = {clean_key_name(k): v for k, v in scan_desc.items()}

        # don't have phi because we need to convert pixels first
        deg_to_rad_coords = {"beta", "theta", "chi"}

        # convert angular attributes to radians
        for coord_name in deg_to_rad_coords:
            if coord_name in attrs:
                try:
                    attrs[coord_name] = float(attrs[coord_name]) * (np.pi / 180)
                except (TypeError, ValueError):
                    pass
            if coord_name in scan_desc:
                try:
                    scan_desc[coord_name] = float(scan_desc[coord_name]) * (np.pi / 180)
                except (TypeError, ValueError):
                    pass

        data_vars = {}

        all_names = hdu.columns.names
        n_spectra = len([n for n in all_names if "Fixed_Spectra" in n or "Swept_Spectra" in n])
        for column_name in hdu.columns.names:
            # we skip some fixed set of the columns, such as the one dimensional axes, as well as things that are too
            # tricky to load at the moment, like the microscope images from MAESTRO
            should_skip = False
            if column_name in self.SKIP_COLUMN_NAMES:
                should_skip = True

            for formula in self.SKIP_COLUMN_FORMULAS:
                if formula(column_name):
                    should_skip = True

            if should_skip:
                continue

            # the hemisphere axis is handled below
            dimension_for_column = dimensions[column_name]
            column_shape = real_spectrum_shape[column_name]

            column_display = self.PREPPED_COLUMN_NAMES.get(column_name, column_name)
            if "Fixed_Spectra" in column_display:
                if n_spectra == 1:
                    column_display = "spectrum"
                else:
                    column_display = "spectrum" + "-" + column_display.split("Fixed_Spectra")[1]

            if "Swept_Spectra" in column_display:
                if n_spectra == 1:
                    column_display = "spectrum"
                else:
                    column_display = "spectrum" + "-" + column_display.split("Swept_Spectra")[1]

            # sometimes if a scan is terminated early it can happen that the sizes do not match the expected value
            # as an example, if a beta map is supposed to have 401 slices, it might end up having only 260 if it were
            # terminated early
            # If we are confident in our parsing code above, we can handle this case and take a subset of the coords
            # so that the data matches
            try:
                resized_data = hdu.data.columns[column_name].array.reshape(column_shape)
            except ValueError:
                # if we could not resize appropriately, we will try to reify the shapes together
                rest_column_shape = column_shape[1:]
                n_per_slice = int(np.prod(rest_column_shape))
                total_shape = hdu.data.columns[column_name].array.shape
                total_n = np.prod(total_shape)

                n_slices = total_n // n_per_slice
                # if this isn't true, we can't recover
                data_for_resize = hdu.data.columns[column_name].array
                if total_n // n_per_slice != total_n / n_per_slice:
                    # the last slice was in the middle of writing when something hit the fan
                    # we need to infer how much of the data to read, and then repeat the above
                    # we need to cut the data

                    # This can happen when the labview crashes during data collection,
                    # we use column_shape[1] because of the row order that is used in the FITS file
                    data_for_resize = data_for_resize[
                        0 : (total_n // n_per_slice) * column_shape[1]
                    ]
                    warnings.warn(
                        "Column {} was in the middle of slice when DAQ stopped. Throwing out incomplete slice...".format(
                            column_name
                        )
                    )

                column_shape = list(column_shape)
                column_shape[0] = n_slices

                try:
                    resized_data = data_for_resize.reshape(column_shape)
                except Exception:
                    # sometimes for whatever reason FITS errors and cannot read the data
                    continue

                # we also need to adjust the coordinates
                altered_dimension = dimension_for_column[0]
                built_coords[altered_dimension] = built_coords[altered_dimension][:n_slices]

            data_vars[column_display] = xr.DataArray(
                resized_data,
                coords={k: c for k, c in built_coords.items() if k in dimension_for_column},
                dims=dimension_for_column,
                attrs=attrs,
            )

        def prep_spectrum(data: xr.DataArray):
            # don't do center pixel inference because the main chamber
            # at least consistently records the offset from the edge
            # of the recorded window
            if "pixel" in data.coords:
                phi_axis = (
                    data.coords["pixel"].values * arpes.constants.SPECTROMETER_MC["rad_per_pixel"]
                )

                if "pixel" in data.coords:
                    data = data.rename(pixel="phi")

                data = data.assign_coords(phi=phi_axis)

            # Always attach provenance
            provenance_from_file(
                data,
                frame_path,
                {
                    "what": "Loaded MC dataset from FITS.",
                    "by": "load_MC",
                },
            )

            return data

        if "spectrum" in data_vars:
            data_vars["spectrum"] = prep_spectrum(data_vars["spectrum"])

        # adjust angular coordinates
        built_coords = {
            k: c * (np.pi / 180) if k in deg_to_rad_coords else c for k, c in built_coords.items()
        }

        return xr.Dataset(
            {
                "safe-{}".format(name) if name in data_var.coords else name: data_var
                for name, data_var in data_vars.items()
            },
            attrs={**scan_desc, "name": primary_dataset_name},
        )


class SynchrotronEndstation(EndstationBase):
    """Base class code for ARPES setups at synchrotrons.

    Synchrotron endstations have somewhat complicated light source metadata.
    This stub exists to attach commonalities, such as a resolution table which
    can be interpolated into to retrieve the x-ray linewidth at the
    experimental settings. Additionally, subclassing this is used in resolution
    calculations to signal that such a resolution lookup is required.
    """

    RESOLUTION_TABLE = None


class HemisphericalEndstation(EndstationBase):
    """Base class code for ARPES setups using hemispheres.

    An endstation definition for a hemispherical analyzer should include
    everything needed to determine energy + k resolution, angle conversion,
    and ideally correction databases for dead pixels + detector nonlinearity
    information
    """

    ANALYZER_INFORMATION = None
    SLIT_ORIENTATION = None
    PIXELS_PER_DEG = None


def endstation_from_alias(alias: str) -> type:
    """Lookup the data loading class from an alias."""
    return _ENDSTATION_ALIASES[alias]


def endstation_name_from_alias(alias) -> str:
    """Lookup the data loading principal location from an alias."""
    return endstation_from_alias(alias).PRINCIPAL_NAME


def add_endstation(endstation_cls: type) -> None:
    """Registers a data loading plugin (Endstation class) together with its aliases.

    You can use this to add a plugin after the original search if it is defined in another
    module or in a notebook.
    """
    assert endstation_cls.PRINCIPAL_NAME is not None
    for alias in endstation_cls.ALIASES:
        if alias in _ENDSTATION_ALIASES:
            continue

        _ENDSTATION_ALIASES[alias] = endstation_cls

    _ENDSTATION_ALIASES[endstation_cls.PRINCIPAL_NAME] = endstation_cls


def resolve_endstation(retry=True, **kwargs) -> type:
    """Tries to determine which plugin to use for loading a piece of data.

    Args:
        retry: Whether to attempt to reload plugins and try again after failure.
          This is used as an import guard basiscally in case the user imported things
          very strangely to ensure plugins are loaded.
        kwargs: Contains the actual information required to identify the scan.

    Returns:
        The loading plugin that should be used for the data.
    """
    endstation_name = case_insensitive_get(
        kwargs, "location", case_insensitive_get(kwargs, "endstation")
    )

    # check if the user actually provided a plugin
    if isinstance(endstation_name, type):
        return endstation_name

    if endstation_name is None:
        warnings.warn("Endstation not provided. Using `fallback` plugin.")
        endstation_name = "fallback"

    try:
        return endstation_from_alias(endstation_name)
    except KeyError:
        if retry:
            import arpes.config  # pylint: disable=redefined-outer-name

            arpes.config.load_plugins()
            return resolve_endstation(retry=False, **kwargs)
        else:
            raise ValueError(
                "Could not identify endstation. "
                "Did you set the endstation or location? Find a description of the available options "
                "in the endstations module."
            )


def load_scan(scan_desc: Dict[str, str], retry=True, **kwargs: Any) -> xr.Dataset:
    """Resolves a plugin and delegates loading a scan.

    This is used interally by `load_data` and should not be invoked directly
    by users.

    Determines which data loading class is appropriate for the data,
    shuffles a bit of metadata, and calls the .load function on the
    retrieved class to start the data loading process.

    Args:
        scan_desc: Information identifying the scan, typically a scan number or full path.
        retry: Used to attempt a reload of plugins and subsequent data load attempt.
        kwargs:

    Returns:
        Loaded and normalized ARPES scan data.
    """
    note = scan_desc.get("note", scan_desc)
    full_note = copy.deepcopy(scan_desc)
    full_note.update(note)

    endstation_cls = resolve_endstation(retry=retry, **full_note)

    key = "file" if "file" in scan_desc else "path"

    file = scan_desc[key]

    try:
        file = int(file)
        file = endstation_cls.find_first_file(file, scan_desc)
        scan_desc[key] = file
    except ValueError:
        pass

    return endstation_cls().load(scan_desc, **kwargs)
