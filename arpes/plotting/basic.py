import pandas as pd
from arpes.utilities import default_dataset
from arpes.preparation import normalize_dim
from arpes.io import load_dataset
from arpes.pipelines import convert_scan_to_kspace

__all__ = ['make_reference_plots']

def make_reference_plots(df: pd.DataFrame=None, with_kspace=False):
    if df is None:
        df = default_dataset()

    df = df[df.spectrum_type != 'xps_spectrum']

    # Make scans indicating cut locations
    for index, row in df.iterrows():
        scan = load_dataset(row.id, df)
        scan.S.reference_plot(out=True, use_id=False)

        if scan.S.spectrum_type == 'spectrum':
            # Also go and make a normalized version
            normed = normalize_dim(scan, 'phi')
            normed.S.reference_plot(out=True, use_id=False, pattern='{}_norm_phi.png')

            if with_kspace:
                kspace_converted = convert_scan_to_kspace(scan)
                kspace_converted.S.reference_plot(out=True, use_id=False, pattern='k_{}.png')

                normed_k = normalize_dim(kspace_converted, 'kp')
                normed_k.S.reference_plot(out=True, use_id=False, pattern='k_{}_norm_kp.png')