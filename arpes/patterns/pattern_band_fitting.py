from .pattern_imports import *
from arpes.fits import *
from arpes.models.band import *

def band_plot():
    test = {}
    def make_band_plot(name, scan, region_name, bands, fig, axes, out=None, color=None):
        x = bands[0].coords[bands[0].dims[0]]
        handles = []
        for band in bands:
            offset = float(band.center.sel(delay=slice(None, scan.S.t0 - 0.2)).mean())
            item = (band.center - offset).plot()
            item[0].set_color(color)
            item[0].set_label('Test')
            handles.append(item[0])
            plt.fill_between(x, band.center.values - band.center_stderr - offset, band.center.values + band.center_stderr - offset, alpha=0.25, color=color)

        axes.axvspan(axes.get_xlim()[0], scan.S.t0, alpha=0.05, color='black')
        axes.set_xlim(np.min(x), np.max(x))
        axes.axvline(x=scan.S.t0, color='black', alpha=0.3, linestyle='--')
        axes.set_xlabel(r'\textbf{Delay} (ps)')
        axes.set_ylabel(r'$\Delta \textbf{k}$')
        axes.set_title('{}'.format(name), fontsize=14)

        if out is not None:
            fig.savefig(os.path.join(FIGURE_PATH, out), dpi=400)

    window_offset = +0.04
    kp_offset = +0.08
    test_regions = {
        'hole_pocket_lip': [{
            'eV': slice(-0.22 + window_offset, -0.19 + window_offset),
            'kp': slice(0.15 + kp_offset, 0.25 + kp_offset),
        }, 'mdc', [
            # BackgroundBand('background'),
            VoigtBand('two'),
        ]],
        'electron_pocket_lip': [{
            'eV': slice(-0.02 + window_offset, 0.02 + window_offset),
            'kp': slice(0.18 + kp_offset, 0.25 + kp_offset),
        }, 'mdc', [
            BackgroundBand('background'),
        ]],
        'above_eF': [{
            'eV': slice(0.04 + window_offset, 0.15 + window_offset),
            'kp': slice(0.1, 0.3),
        }, 'mdc', [
            VoigtBand('background'),
            VoigtBand('background2'),
        ]],
        'mu': [{
            'eV': slice(-0.2, None),
            'kp': slice(0.05, 0.12),
        }, 'edc', [
            {
                'params': {
                    'lin_bkg': {'max': 0.1, 'min': -0.1, 'value': 0},
                    'const_bkg': {'max': 0.1, 'min': -0.1, 'value': 0},
                    'center': {'min': -0.15, 'max': 0.05, 'value': -0.05},
                    'sigma': {'value': 0.1, 'min': 0.1, 'max': 0.2},
                    'amplitude': {'max': 0.02, 'min': 0.01}
                },
                'band': FermiEdgeBand('mu'),
            }
        ]]
    }

    del test_regions['mu']

    for region_name, region in test_regions.items():
        if region_name == 'above_eF':
            continue
        region, fit_mode, model = region
        fig, axes = plt.subplots(figsize=(6, 4))
        idx = -1
        colors = ['purple', 'red']
        for scan, name in zfs[-4:-2]:
            idx += 1
            print(name, region_name)
            if fit_mode == 'mdc':
                fit_region = scan.sel(**region).sum('eV')
            else:
                fit_region = scan.sel(**region).sum('kp')
            fit_region = boxcar_filter_arr(fit_region, {'kp': 0.02}, n=2)
            results, bands, residual = fit_bands(fit_region, model, direction=fit_mode)

            make_band_plot(name, scan, region_name, bands, fig, axes,
                           out='{}_{}_{}.png'.format(region_name, kp_offset, window_offset), color=colors[idx])
        plt.show()

