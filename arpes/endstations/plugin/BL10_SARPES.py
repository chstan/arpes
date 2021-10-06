"""The Spin-ARPES setup at beamline 10.0.1.2 of the Advanced Light Source."""
from pathlib import Path

import numpy as np

import typing
import xarray as xr
from arpes.endstations import HemisphericalEndstation, SESEndstation, SynchrotronEndstation
import arpes.xarray_extensions

__all__ = ["BL10012SARPESEndstation"]


class BL10012SARPESEndstation(SynchrotronEndstation, HemisphericalEndstation, SESEndstation):
    """The Spin-ARPES setup at beamline 10.0.1.2 at the Advanced Light Source."""

    PRINCIPAL_NAME = "ALS-BL10-SARPES"
    ALIASES = [
        "BL10-SARPES",
    ]

    _TOLERATED_EXTENSIONS = {
        ".pxt",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w+]+_{}_S[0-9][0-9][0-9]$",
        r"[\-a-zA-Z0-9_\w+]+_{}_R[0-9][0-9][0-9]$",
        r"[\-a-zA-Z0-9_\w+]+_[0]+{}_S[0-9][0-9][0-9]$",
        r"[\-a-zA-Z0-9_\w+]+_[0]+{}_R[0-9][0-9][0-9]$",
        # more generic
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )

    RENAME_KEYS = {
        # TODO Kayla or another user should add these
        # Look at merlin.py for details
    }

    MERGE_ATTRS = {
        # TODO Kayla or another user should add these
        # Look at merlin.py for details
    }

    ATTR_TRANSFORMS = {
        # TODO Kayla or another user should add these
        # Look at merlin.py for details
    }
    SPIN_RENAMINGS = {
        "W": "eV",
        "White_spin": "white_spin_unknown",
        "White_spin_White_Xplus": "spectrum_spin_x_up",
        "White_spin_White_Xminus": "spectrum_spin_x_down",
        "White_spin_White_Yplus": "spectrum_spin_y_up",
        "White_spin_White_Yminus": "spectrum_spin_y_down",
        "White_spin_White_Zplus": "spectrum_spin_z_up",
        "White_spin_White_Zminus": "spectrum_spin_z_down",
    }

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """Loads all regions for a single .pxt frame, and perform per-frame normalization."""
        from arpes.load_pxt import read_single_pxt, find_ses_files_associated

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))

        p = Path(original_data_loc)

        # find files with same name stem, indexed in format R###
        regions = find_ses_files_associated(p, separator="R")

        if len(regions) == 1:
            pxt_data = read_single_pxt(frame_path, allow_multiple=True)
            pxt_data = pxt_data.rename(
                {k: v for k, v in self.SPIN_RENAMINGS.items() if k in pxt_data}
            )
            return pxt_data
        else:
            # need to merge several different detector 'regions' in the same scan
            region_files = [self.load_single_region(region_path) for region_path in regions]

            # can they share their energy axes?
            all_same_energy = True
            for reg in region_files[1:]:
                dim = "eV" + reg.attrs["Rnum"]
                all_same_energy = all_same_energy and np.array_equal(
                    region_files[0].coords["eV000"], reg.coords[dim]
                )

            if all_same_energy:
                for i, reg in enumerate(region_files):
                    dim = "eV" + reg.attrs["Rnum"]
                    region_files[i] = reg.rename({dim: "eV"})
            else:
                pass

            return self.concatenate_frames(region_files, scan_desc=scan_desc)

    def load_single_region(self, region_path: str = None, scan_desc: dict = None, **kwargs):
        """Loads a single region for multi-region scans."""
        import os
        from arpes.load_pxt import read_single_pxt

        name, _ = os.path.splitext(region_path)
        num = name[-3:]

        pxt_data = read_single_pxt(region_path, allow_multiple=True)
        pxt_data = pxt_data.rename({"eV": "eV" + num})
        pxt_data.attrs["Rnum"] = num
        pxt_data.attrs["alpha"] = np.pi / 2.0

        pxt_data = pxt_data.rename({k: f"{k}{num}" for k in pxt_data.data_vars})
        return pxt_data

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        """Performs final data normalization for MERLIN data.

        Additional steps we perform here are:

        1. We attach the slit information for the R8000 used on MERLIN.
        2. We normalize the undulator polarization from the sentinel values
          recorded by the beamline.
        3. We convert angle units to radians.

        Args:
            data: The input data
            scan_desc: Originating load parameters

        Returns:
            Processed copy of the data
        """
        ls = [data] + data.S.spectra

        deg_to_rad_coords = {"theta", "phi", "beta", "chi", "psi"}
        deg_to_rad_attrs = {"theta", "beta", "chi", "psi", "alpha"}

        for c in deg_to_rad_coords:
            if c in data.dims:
                data.coords[c] = data.coords[c] * np.pi / 180

        for angle_attr in deg_to_rad_attrs:
            for l in ls:
                if angle_attr in l.attrs:
                    l.attrs[angle_attr] = float(l.attrs[angle_attr]) * np.pi / 180

        data.attrs["alpha"] = np.pi / 2
        data.attrs["psi"] = 0
        for s in data.S.spectra:
            s.attrs["alpha"] = np.pi / 2
            s.attrs["psi"] = 0

        # TODO Conrad think more about why sometimes individual attrs don't make it onto
        # .spectrum.attrs, for now just paste them over
        necessary_coord_names = {"theta", "beta", "chi", "phi"}
        ls = data.S.spectra
        for l in ls:
            for cname in necessary_coord_names:
                if cname not in l.attrs and cname not in l.coords and cname in data.attrs:
                    l.attrs[cname] = data.attrs[cname]

        return super().postprocess_final(data, scan_desc)
