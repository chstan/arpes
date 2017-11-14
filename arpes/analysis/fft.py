# TODO note that there is a slight bug in the inverse Fourier transform here where the coords are sometimes rolled

import warnings

import dask as dsar
import numpy as np
import pandas as pd
import xarray as xr
import xrft
from xrft.xrft import _hanning

from arpes.provenance import provenance

# Don't export the xrft additions we monkey patch them onto that module directly.
__all__ = ('fft_filter',)

def fft_filter(data: xr.DataArray, stops):
    """
    Applies a brick wall filter at region in ``stops`` in the Fourier transform of data. Use with care.
    :param data:
    :param stops:
    :return:
    """

    # This won't tolerate inverse inverse filtering ;)
    kdata = xrft.dft(data)

    # this will produce a fair amount of ringing, Scipy isn't very clear about how to butterworth filter in Nd
    for stop in stops:
        kstop = {'freq_' + k if 'freq_' not in k else k: v for k, v in stop.items()} # be nice
        kdata.loc[kstop] = 0

    kkdata = xrft.idft(kdata)
    kkdata.values = np.real(kkdata.values)
    kkdata.values = kkdata.values - np.min(kkdata.values) + np.mean(data.values)

    filtered_arr = xr.DataArray(
        kkdata,
        data.coords,
        data.dims,
        attrs=data.attrs.copy()
    )

    if 'id' in filtered_arr:
        del filtered_arr.attrs['id']

        provenance(filtered_arr, data, {
            'what': 'Apply a filter in frequency space by brick walling coordinate regions.',
            'by': 'fft_filter',
            'stops': stops,
        })

    return filtered_arr


def _spacing(da, dims):
    """
    Verify correct spacing and return the spacing for each axis
    :param da:
    :return:
    """
    delta_x = []
    for d in dims:
        coord = da[d]
        diff = np.diff(coord)
        if pd.core.common.is_timedelta64_dtype(diff):
            # convert to seconds so we get hertz
            diff = diff.astype('timedelta64[s]').astype('f8')
        delta = diff[0]
        if not np.allclose(diff, diff[0]):
            raise ValueError("Can't take Fourier transform because"
                             "coodinate %s is not evenly spaced" % d)
        delta_x.append(delta)

    return delta_x


def rdft(da, dim=None, shift=True, remove_mean=True, window=False):
    """
    Perform real discrete Fourier transform of xarray data-array `da` along the
    specified dimensions.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
    dim : list (optional)
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool (optional)
        Whether to shift the fft output.
    remove_mean : bool (optional)
        If `True`, the mean across the transform dimensions will be subtracted
        before calculating the Fourier transform.

    Returns
    -------
    daft : `xarray.DataArray`
        The output of the Fourier transformation, with appropriate dimensions.
    """
    if np.isnan(da.values).any():
        raise ValueError("Data cannot take Nans")

    if dim is None:
        dim = da.dims

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]
    delta_x = _spacing(da, dim)

    # calculate frequencies from coordinates
    k = [ np.fft.rfftfreq(Nx, dx) for (Nx, dx) in zip(N, delta_x) ]

    if remove_mean:
        da = da - da.mean(dim=dim)

    if window:
        da = _hanning(da, N)

    # the hard work
    #f = np.fft.rfftn(da.values, axes=axis_num)
    # need special path for dask
    # is this the best way to check for dask?
    data = da.data
    if hasattr(data, 'dask'):
        assert len(axis_num)==1
        f = dsar.fft.rfft(data, axis=axis_num[0])
    else:
        f = np.fft.rfftn(data, axes=axis_num)

    if shift:
        f = np.fft.fftshift(f, axes=axis_num)
        k = [np.fft.fftshift(l) for l in k]

    # set up new coordinates for dataarray
    prefix = 'freq_'
    k_names = [prefix + d for d in dim]
    k_coords = {key: val for (key,val) in zip(k_names, k)}

    newdims = list(da.dims)
    for anum, d in zip(axis_num, dim):
        newdims[anum] = prefix + d

    newcoords = {}
    for d in newdims:
        if d in da.coords:
            newcoords[d] = da.coords[d]
        else:
            newcoords[d] = k_coords[d]

    dk = [l[1] - l[0] for l in k]

    new_attrs = da.attrs.copy()
    new_attrs['N'] = N
    return xr.DataArray(f, dims=newdims, coords=newcoords, attrs=new_attrs)



def irdft(da, dim=None, shift=True, window=False):
    """
    Perform inverse real discrete Fourier transform of xarray data-array `da` along the
    specified dimensions.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
    dim : list (optional)
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool (optional)
        Whether to shift the fft output.

    Returns
    -------
    daft : `xarray.DataArray`
        The output of the Fourier transformation, with appropriate dimensions.
    """
    if np.isnan(da.values).any():
        raise ValueError("Data cannot take NaNs")

    if dim is None:
        dim = da.dims

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]
    delta_x = _spacing(da, dim)

    # calculate frequencies from coordinates
    k = [ np.fft.fftfreq(Nx, dx) for (Nx, dx) in zip(N, delta_x) ]

    if window:
        da = _hanning(da, N)

    # the hard work
    #f = np.fft.irfftn(da.values, axes=axis_num)
    # need special path for dask
    # is this the best way to check for dask?
    data = da.data
    if shift:
        data = np.fft.ifftshift(data, axes=axis_num)
        k = [np.fft.ifftshift(l) for l in k]

    if hasattr(data, 'dask'):
        assert len(axis_num)==1
        f = dsar.fft.irfft(data, axis=axis_num[0], s=da.attrs.get('N'))
    else:
        f = np.fft.irfftn(data, axes=axis_num, s=da.attrs.get('N'))

    # set up new coordinates for dataarray
    prefix = 'freq_'
    if any(prefix in d for d in da.dims):
        # came from a FFT we performed
        k_names = [d.split(prefix)[-1] for d in dim]
    else:
        k_names = [prefix + d for d in dim]

    k_coords = {key: val for (key,val) in zip(k_names, k)}

    newdims = list(da.dims)
    for anum, d in zip(axis_num, k_names):
        newdims[anum] = d

    newcoords = {}
    for d in newdims:
        if d in da.coords:
            newcoords[d] = da.coords[d]
        else:
            newcoords[d] = k_coords[d]

    dk = [l[1] - l[0] for l in k]

    new_attrs = da.attrs.copy()
    new_attrs['N'] = N
    return xr.DataArray(f, dims=newdims, coords=newcoords, attrs=new_attrs)


def idft(da, dim=None, shift=True, window=False):
    """
    Perform inverse real discrete Fourier transform of xarray data-array `da` along the
    specified dimensions.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
    dim : list (optional)
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool (optional)
        Whether to shift the fft output.

    Returns
    -------
    daft : `xarray.DataArray`
        The output of the Fourier transformation, with appropriate dimensions.
    """
    warnings.warn('This is a little buggy, it does not produce an output with monotonic coords for some reason.')
    if np.isnan(da.values).any():
        raise ValueError("Data cannot take NaNs")

    if dim is None:
        dim = da.dims

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]
    delta_x = _spacing(da, dim)

    # calculate frequencies from coordinates
    k = [ np.fft.fftfreq(Nx, dx) for (Nx, dx) in zip(N, delta_x) ]

    if window:
        da = _hanning(da, N)

    # the hard work
    #f = np.fft.ifftn(da.values, axes=axis_num)
    # need special path for dask
    # is this the best way to check for dask?
    data = da.data
    if shift:
        data = np.fft.ifftshift(data, axes=axis_num)
        k = [np.fft.ifftshift(l) for l in k]

    if hasattr(data, 'dask'):
        assert len(axis_num)==1
        f = dsar.fft.ifft(data, axis=axis_num[0], s=da.attrs.get('N'))
    else:
        f = np.fft.ifftn(data, axes=axis_num, s=da.attrs.get('N'))

    # set up new coordinates for dataarray
    prefix = 'freq_'
    if any(prefix in d for d in da.dims):
        # came from a FFT we performed
        k_names = [d.split(prefix)[-1] for d in dim]
    else:
        k_names = [prefix + d for d in dim]

    k_coords = {key: val for (key,val) in zip(k_names, k)}

    newdims = list(da.dims)
    for anum, d in zip(axis_num, k_names):
        newdims[anum] = d

    newcoords = {}
    for d in newdims:
        if d in da.coords:
            newcoords[d] = da.coords[d]
        else:
            newcoords[d] = k_coords[d]

    dk = [l[1] - l[0] for l in k]

    new_attrs = da.attrs.copy()
    new_attrs['N'] = N
    return xr.DataArray(f, dims=newdims, coords=newcoords, attrs=new_attrs)


# Monkey Patch
xrft.rdft = rdft
xrft.irdft = irdft
xrft.idft = idft

