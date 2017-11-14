import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage import feature

__all__ = ('infer_center_pixel',)

def infer_center_pixel(arr: xr.DataArray):
    near_ef = arr.sel(eV=slice(-0.1, 0)).sum([d for d in arr.dims if d not in ['pixel']])
    embed_size = 20
    embedded = np.ndarray(shape=[embed_size] + list(near_ef.values.shape))
    embedded[:] = near_ef.values
    embedded = ndi.gaussian_filter(embedded, embed_size / 3)

    edges = feature.canny(embedded, sigma=embed_size / 5, use_quantiles=True,
                          low_threshold=0.2) * 1
    edges = np.where(edges[int(embed_size / 2)] == 1)

    return float((np.max(edges) + np.min(edges)) / 2 + np.min(arr.coords['pixel']))