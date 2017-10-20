import holoviews as hv
import matplotlib.pyplot as plt
import xarray as xr


def band_path(band):
    return hv.Path([band.center.values, band.coords[band.dims[0]].values])

def plot_dispersion_holoview(spectrum: xr.DataArray, bands):
    image = hv.Image(spectrum)
    band = band_path(bands[1])
    return band

def plot_dispersion(spectrum: xr.DataArray, bands):
    ax = spectrum.plot()

    for band in bands:
        plt.scatter(band.center.values, band.coords[band.dims[0]].values)

    return ax