"""implements data loading for ANTARES at SOLEIL."""
from collections import Counter
import warnings

import h5py
import numpy as np

import xarray as xr
from arpes.endstations import HemisphericalEndstation, SingleFileEndstation, SynchrotronEndstation
from arpes.endstations.nexus_utils import (
    AttrTarget,
    CoordTarget,
    DebugTarget,
    read_data_attributes_from,
    read_data_attributes_from_tree,
)
from arpes.preparation import disambiguate_coordinates

__all__ = ("ANTARESEndstation",)

MONO_READ_TREE = {
    "energy": CoordTarget("hv"),
    "exitSlitAperature": AttrTarget("exit_slit_aperature"),
    "resolution": AttrTarget("resolution"),
    "currentGratingName": AttrTarget("current_grating_name"),
    "currentSlotName": AttrTarget("current_slot_name"),
}

USER_INFO_READ_TREE = {
    "email": AttrTarget("user_email"),
    "address": AttrTarget("user_address"),
    "affiliation": AttrTarget("user_affiliation"),
    "name": AttrTarget("user_name"),
    "telephone_number": AttrTarget("user_telephone_number"),
}

READ_TREE = {
    "ANTARES": {"Monochromator": MONO_READ_TREE},
    "User": USER_INFO_READ_TREE,
    "comment_conditions": AttrTarget("comment_conditions"),
    "experimental_frame": AttrTarget("experimental_frame"),
    "start_time": AttrTarget("start_time"),
}

MBS_TREE = {
    "Frames": AttrTarget("frames"),
    "LensMode": AttrTarget("lens_mode"),
    "PASSENERGY": AttrTarget("pass_energy"),
    "DeflX": CoordTarget("psi"),
    "DeflY": CoordTarget("defl_y"),
    "CenterKE": AttrTarget("center_ke"),
    "StepSize": AttrTarget("mbs_step_size"),
    "StartX": AttrTarget("mbs_start_x"),
    "StartY": AttrTarget("mbs_start_y"),
    "EndX": AttrTarget("mbs_end_x"),
    "EndY": AttrTarget("mbs_end_y"),
    "StartKE": AttrTarget("mbs_start_ke"),
    "NoSlices": AttrTarget("mbs_no_slices"),
    "NoScans": AttrTarget("mbs_no_scans"),
}


def parse_axis_name_from_long_name(name, keep_segments=1, separator="_"):
    segments = name.split("/")[-keep_segments:]
    segments = [s.replace("'", "") for s in segments]
    return separator.join(segments)


def infer_scan_type_from_data(group):
    """Determines the scan type for NeXuS format data.

    Because ANTARES stores every possible data type in the NeXuS file format, zeroing information that is
    not used, we have to determine which data folder to use on the basis of what kind of scan was done.
    """
    scan_name = str(group["scan_config"]["name"][()])

    if "DeflX" in scan_name:
        # Fermi Surface, might need to be more robust
        return "data_09"

    if "Scan2D_MBS" in scan_name:
        # two piezo or two DOF image scan
        return "data_12"

    raise NotImplementedError(scan_name)


class ANTARESEndstation(HemisphericalEndstation, SynchrotronEndstation, SingleFileEndstation):
    """Implements data loading for ANTARES at SOLEIL.

    There's not too much metadata here except what comes with the analyzer settings.
    """

    PRINCIPAL_NAME = "ANTARES"
    ALIASES = []

    _TOLERATED_EXTENSIONS = {".nxs"}

    RENAME_KEYS = {}

    def load_top_level_scan(self, group, scan_desc: dict = None, spectrum_index=None):
        """Reads a spectrum from the top level group in a NeXuS scan format."""
        dr = self.read_scan_data(group)
        bindings = read_data_attributes_from_tree(group, READ_TREE)

        for binding in bindings:
            binding.write_to_dataarray(dr)

        try:
            mbs_key = [k for k in list(group["ANTARES"].keys()) if "MBSAcquisition" in k][0]
            mbs_group = group["ANTARES"][mbs_key]
            mbs_bindings = read_data_attributes_from_tree(mbs_group, MBS_TREE)
            bindings.extend(mbs_bindings)
        except IndexError:
            pass

        ds = xr.Dataset(dict([["spectrum-{}".format(spectrum_index), dr]]))

        for binding in bindings:
            binding.write_to_dataset(ds)

        return ds

    def get_coords(self, group, scan_name, shape):
        """Extracts coordinates from the actuator header information.

        In the future, this should be modified for data which lacks either a phi or energy axis.
        """
        dims = list(shape)
        data = group["scan_data"]

        # handle actuators
        relaxed_shape = list(shape)
        actuator_list = [k for k in list(data.keys()) if "actuator" in k]
        actuator_long_names = [str(data[act].attrs["long_name"]) for act in actuator_list]
        actuator_names = [parse_axis_name_from_long_name(name) for name in actuator_long_names]

        # This more carefully deduplicates names if they have a common
        # suffix in the long name format.
        keep_segments = 1
        set_names = Counter(actuator_names)
        while len(set_names) != len(actuator_names):
            keep_segments += 1
            actuator_names = [
                name
                if set_names[name] == 1
                else parse_axis_name_from_long_name(actuator_long_names[i], keep_segments)
                for i, name in enumerate(actuator_names)
            ]
            set_names = Counter(actuator_names)

        actuator_list = [data[act][:] for act in actuator_list]

        actuator_dim_order = []
        for act in actuator_list:
            found = relaxed_shape.index(act.shape[-1])
            actuator_dim_order.append(found)
            relaxed_shape[found] = None

        coords = {}

        def take_last(vs):
            while len(vs.shape) > 1:
                vs = vs[0]

            return vs

        for dim_order, name, values in zip(actuator_dim_order, actuator_names, actuator_list):
            name = self.RENAME_KEYS.get(name, name)
            dims[dim_order] = name
            coords[name] = take_last(values)

        # handle standard spectrometer axes, keeping in mind things get stored
        # in different places sometimes for no reasons
        energy_keys = {
            "data_09": (
                "data_01",
                "data_03",
                "data_02",
            ),
            "data_12": (
                "data_04",
                "data_06",
                "data_05",
            ),
        }
        angle_keys = {
            "data_09": (
                "data_04",
                "data_06",
                "data_05",
            ),
            "data_12": (
                "data_07",
                "data_09",
                "data_08",
            ),
        }

        e_keys = energy_keys[scan_name]
        ang_keys = angle_keys[scan_name]
        energy = data[e_keys[0]][0], data[e_keys[1]][0], data[e_keys[2]][0]
        angle = data[ang_keys[0]][0], data[ang_keys[1]][0], data[ang_keys[2]][0]

        def get_first(item):
            if isinstance(item, np.ndarray):
                return item.ravel()[0]

            return item

        def build_axis(low, high, step_size):
            # this might not work out to be the right thing to do, we will see
            low, high, step_size = get_first(low), get_first(high), get_first(step_size)
            est_n = int((high - low) / step_size)

            closest = None
            diff = np.inf
            idx = None
            for i, s in enumerate(shape):
                if closest is None or np.abs(s - est_n) < diff:
                    idx = i
                    diff = np.abs(s - est_n)
                    closest = s

            if diff != 0:
                warnings.warn("Could not identify axis by length.")
            return np.linspace(low, high, closest, endpoint=False), idx

        energy, energy_idx = build_axis(*energy)
        angle, angle_idx = build_axis(*angle)

        dims[energy_idx] = "eV"
        dims[angle_idx] = "phi"
        coords["eV"] = energy
        coords["phi"] = angle * np.pi / 180

        return dims, coords

    def read_scan_data(self, group):
        """Reads the scan data stored in /scan_data/data_{idx} for the appropriate filetype."""
        data_key = infer_scan_type_from_data(group)
        data_group = group["scan_data"][data_key]
        data = data_group[:]

        dims, coords = self.get_coords(group, data_key, shape=data.shape)

        return xr.DataArray(data, coords=coords, dims=dims)

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """Loads a single ANTARES scan.

        Additionally, we try to deduplicate coordinates for multi-region scans here.
        """
        f = h5py.File(frame_path)
        top_level = list(f.keys())

        loaded = [
            self.load_top_level_scan(f[key], scan_desc, spectrum_index=i)
            for i, key in enumerate(top_level)
        ]

        if isinstance(loaded, list) and loaded:
            loaded = disambiguate_coordinates(loaded, ["phi", "eV"])
            loaded = xr.merge(loaded)
        else:
            loaded = loaded[0]
            loaded.rename({"spectrum-1": "spectrum"})

        loaded = loaded.assign_attrs(
            **{self.RENAME_KEYS.get(k, k): v for k, v in loaded.attrs.items()}
        )

        return loaded

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        """Performs final scan postprocessing.

        This mostly consists of unwrapping bytestring attributes, and
        inserting missing default coordinates if they are not provided.
        """

        def check_attrs(s):
            for k in ["psi", "hv", "lens_mode", "pass_energy"]:
                try:
                    if isinstance(
                        s.attrs[k],
                        (
                            np.ndarray,
                            list,
                            tuple,
                        ),
                    ):
                        s.attrs[k] = s.attrs[k][0]
                    elif isinstance(s.attrs[k], bytes):
                        s.attrs[k] = s.attrs[k].decode()
                except (TypeError, KeyError):
                    pass

        ls = [data] + data.S.spectra
        for l in ls:
            check_attrs(l)

        # attempt to determine whether the energy is likely a kinetic energy
        # if so, we will subtract the photon energy
        if "eV" in data.indexes:
            mean_energy = data["eV"].values.mean()
            photon_energy = data.coords.get("hv", 0)

        # TODO fix this
        defaults = {
            "z": 0,
            "x": 0,
            "y": 0,
            "alpha": 0,
            "chi": 0,
            "theta": 0,
            "beta": 0,
            "hv": None,
            "psi": 0,
        }
        for k, v in defaults.items():
            data.attrs[k] = data.attrs.get(k, v)
            for s in data.S.spectra:
                s.attrs[k] = s.attrs.get(k, v)

        return super().postprocess_final(data, scan_desc)
