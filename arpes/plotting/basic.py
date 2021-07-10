"""Reference plots, for preliminary analysis."""
import warnings

import pandas as pd

import xarray as xr
from arpes.io import load_data
from arpes.preparation import normalize_dim
from arpes.utilities.conversion import convert_to_kspace

__all__ = ["make_reference_plots"]


def make_reference_plots(df: pd.DataFrame = None, with_kspace=False):
    """Makes standard reference plots for orienting oneself."""
    try:
        df = df[df.spectrum_type != "xps_spectrum"]
    except TypeError:
        warnings.warn("Unable to filter out XPS files, did you attach spectra type?")

    # Make scans indicating cut locations
    for index, row in df.iterrows():
        try:
            scan = load_data(index)

            if isinstance(scan, xr.Dataset):
                # make plot series normalized by current:
                scan.S.reference_plot(out=True)
            else:
                scan.S.reference_plot(out=True, use_id=False)

                if scan.S.spectrum_type == "spectrum":
                    # Also go and make a normalized version
                    normed = normalize_dim(scan, "phi")
                    normed.S.reference_plot(out=True, use_id=False, pattern="{}_norm_phi.png")

                    if with_kspace:
                        normalized = normalize_dim(scan, "hv")
                        kspace_converted = convert_to_kspace(normalized)
                        kspace_converted.S.reference_plot(
                            out=True, use_id=False, pattern="k_{}.png"
                        )

                        normed_k = normalize_dim(kspace_converted, "kp")
                        normed_k.S.reference_plot(
                            out=True, use_id=False, pattern="k_{}_norm_kp.png"
                        )

        except Exception as e:
            print(str(e))
            warnings.warn("Cannot make plots for {}".format(index))
