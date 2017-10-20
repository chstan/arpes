import holoviews as hv
import xarray as xr

def fermi_surface_slices(arr: xr.DataArray, n_slices=9, ev_per_slice=0.02, bin=0.01, **kwargs):
    slices = []
    for i in range(n_slices):
        high = - ev_per_slice * i
        low = high - bin
        image = hv.Image(arr.sum([d for d in arr.dims if d not in ['polar', 'phi', 'eV', 'kp', 'kx', 'ky']]).sel(
            eV=slice(low, high)).sum('eV'), label='%g eV' % high)

        slices.append(image)
        
    return hv.Layout(slices).cols(3)
