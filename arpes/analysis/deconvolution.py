import numpy as np
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

from arpes.fits.fit_models import gaussian
import xarray as xr

import scipy

from tqdm import tqdm_notebook

__all__ = ('deconvolve_ice','deconvolve_rl','make_psf1d',)

"""
def _convolve(original_data, convolution_kernel):
    if len(convolution_kernel.shape) == 1:
        conv_kern_norm = convolution_kernel / np.sum(convolution_kernel)
        n_points = min(len(original_data), len(conv_kern_norm))
        padding = np.ones(n_points)
        temp = np.concatenate((padding * original_data[0], original_data, padding * original_data[-1]))
        convolved = np.convolve(temp, conv_kern_norm, mode='valid')
        n_offset = int((len(convolved) - n_points) / 2)
        result = (convolved[n_offset:])[:n_points]
    elif len(convolution_kernel.shape) == 2:
        raise NotImplementedError
    return result
"""

def deconvolve_ice(data: DataType,psf,n_iterations=5,deg=None):
    """Deconvolves data by a given point spread function using the iterative convolution extrapolation method.
    
    :param data:
    :param psf:
    :param n_iterations -- the number of convolutions to use for the fit (default 5):
    :param deg -- the degree of the fitting polynominal (default n_iterations-3):
    :return DataArray or numpy.ndarray -- based on input type:
    """
    
    arr = normalize_to_spectrum(data)
    if type(data) is np.ndarray:
        pass
    else:
        arr = arr.values
    
    if deg is None:
        deg = n_iterations - 3
    iteration_steps = list(range(1,n_iterations+1))

    iteration_list = [arr]

    for i in range(n_iterations-1):
        iteration_list.append(scipy.ndimage.convolve(iteration_list[-1],psf))
    iteration_list = np.asarray(iteration_list)

    deconv = arr*0
    for t, series in enumerate(iteration_list.T):
        coefs = np.polyfit(iteration_steps,series,deg=deg)
        poly = np.poly1d(coefs)
        deconv[t] = poly(0)
    
    if type(data) is np.ndarray:
        result = deconv
    else:
        result = normalize_to_spectrum(data).copy(deep=True)
        result.values = deconv
    return result

def deconvolve_rl(data: DataType, psf=None, n_iterations=10, axis=None,
                  sigma=None, mode='reflect', progress=True):
    """Deconvolves data by a given point spread function using the Richardson-Lucy method.
    
    :param data:
    :param psf -- for 1d, if not specified, must specify axis and sigma:
    :param n_iterations -- the number of convolutions to use for the fit (default 50):
    :param axis:
    :param sigma:
    :param mode:
    :param progress:
    :return DataArray or numpy.ndarray -- based on input type:
    """
    
    arr = normalize_to_spectrum(data)
    
    if psf is None and axis is not None and sigma is not None:
        # if no psf is provided and we have the information to make a 1d one
        # note: this assumes gaussian psf
        psf = make_psf1d(data=arr,dim=axis,sigma=sigma)
    
    if len(data.dims) > 1:
        if axis is not None:
            # perform one-dimensional deconvolution of multidimensional data
            
            # support for progress bars
            wrap_progress = lambda x, *args, **kwargs: x
            if progress:
                wrap_progress = lambda x, *args, **kwargs: tqdm_notebook(x, *args, **kwargs)
            
            # dimensions over which to iterate
            other_dim = list(data.dims)
            other_dim.remove(axis)
            
            if len(other_dim) == 1:
                # two-dimensional data
                other_dim = other_dim[0]
                result = arr.copy(deep=True).transpose(other_dim,axis)  # not sure why the dims only seems to work in this order. seems like I should be able to swap it to (axis,other_dim) and also change the data collection to result[x_ind,y_ind], but this gave different results
                
                for i,(od,iteration) in wrap_progress(enumerate(arr.T.iterate_axis(other_dim)),desc="Iterating " + other_dim,total=len(arr[other_dim])):  # TODO tidy this gross-looking loop
                    # indices of data being deconvolved
                    x_ind = xr.DataArray(list(range(len(arr[axis]))),dims=[axis])
                    y_ind = xr.DataArray([i] * len(x_ind),dims=[other_dim])
                    # perform deconvolution on this one-dimensional piece
                    deconv = deconvolve_rl(data=iteration,psf=psf,n_iterations=n_iterations,axis=None,mode=mode)
                    # build results out of these pieces
                    result[y_ind,x_ind] = deconv.values
            elif len(other_dim) == 2:
                # three-dimensional data
                result = arr.copy(deep=True).transpose(*other_dim,axis)  # not sure why the dims only seems to work in this order. seems like I should be able to swap it to (axis,*other_dim) and also change the data collection to result[x_ind,y_ind,z_ind], but this gave different results
                for i,(od0,iteration0) in wrap_progress(enumerate(arr.T.iterate_axis(other_dim[0])),desc="Iterating " + other_dim[0],total=len(arr[other_dim[0]])):  # TODO tidy this gross-looking loop
                    for j,(od1,iteration1) in wrap_progress(enumerate(iteration0.T.iterate_axis(other_dim[1])),desc="Iterating " + other_dim[1],total=len(arr[other_dim[1]]),leave=False):  # TODO tidy this gross-looking loop
                        # indices of data being deconvolved
                        x_ind = xr.DataArray(list(range(len(arr[axis]))),dims=[axis])
                        y_ind = xr.DataArray([i] * len(x_ind),dims=[other_dim[0]])
                        z_ind = xr.DataArray([j] * len(x_ind),dims=[other_dim[1]])
                        # perform deconvolution on this one-dimensional piece
                        deconv = deconvolve_rl(data=iteration1,psf=psf,n_iterations=n_iterations,axis=None,mode=mode)
                        # build results out of these pieces
                        result[y_ind,z_ind,x_ind] = deconv.values
            elif len(other_dim) >= 3:
                # four- or higher-dimensional data
                # TODO find way to compactify the different dimensionalities rather than having separate code
                raise NotImplementedError("high-dimensional data not yet supported")
                
        elif axis is None:
            # crude attempt to perform multidimensional deconvolution. not clear if this is currently working
            # TODO may be able to do this as a sequence of one-dimensional deconvolutions, assuming that the psf is separable (which I think it should be, if we assume it is a multivariate gaussian with principle axes aligned with the dimensions)
            raise NotImplementedError("multi-dimensional convolutions not yet supported")
            
            if not isinstance(arr, np.ndarray):
                arr = arr.values

            u = [arr]

            for i in range(n_iterations):
                c = scipy.ndimage.convolve(u[-1],psf,mode=mode)
                u.append(u[-1] * scipy.ndimage.convolve(arr/c,np.flip(psf,None),mode=mode))  # careful about which axis (axes) to flip here...! need to explicitly specify for some versions of numpy

            result = u[-1]
    else:
        if type(arr) is not np.ndarray:
            arr = arr.values

        u = [arr]

        for i in range(n_iterations):
            c = scipy.ndimage.convolve(u[-1],psf,mode=mode)
            u.append(u[-1] * scipy.ndimage.convolve(arr/c,np.flip(psf,0),mode=mode))  # not yet tested to ensure flip correct for asymmetric psf
            # note: need to explicitly specify axis number in np.flip in lower versions of numpy

        if type(data) is np.ndarray:
            result = u[-1].copy()
        else:
            result = normalize_to_spectrum(data).copy(deep=True)
            result.values = u[-1]
            
    try:
        result = result.transpose(*arr.dims)
    except:
        pass
            
    return result

def make_psf1d(data: DataType,dim,sigma):
    """Produces a 1-dimensional gaussian point spread function for use in deconvolve_rl.
    
    :param data:
    :param dim:
    :param sigma:
    :return DataArray:
    """
    
    
    arr = normalize_to_spectrum(data)
    dims = arr.dims
    
    psf = arr.copy(deep=True) * 0 + 1
    
    other_dims = list(arr.dims)
    other_dims.remove(dim)

    for od in other_dims:
        psf = psf[{od:0}]

    psf = psf * gaussian(psf.coords[dim],np.mean(psf.coords[dim]),sigma)

    return psf

def make_psf(data: DataType,sigmas):
    """Not yet operational; produces an n-dimensional gaussian point spread function for use in deconvolve_rl.
    
    :param data:
    :param dim:
    :param sigma:
    :return DataArray:
    """
    
    raise NotImplementedError
    
    arr = normalize_to_spectrum(data)
    dims = arr.dims
    
    psf = arr.copy(deep=True) * 0 + 1
    
    for dim in dims:
        other_dims = list(arr.dims)
        other_dims.remove(dim)
        
        psf1d = arr.copy(deep=True) * 0 + 1
        for od in other_dims:
            psf1d = psf1d[{od:0}]
        
        if sigmas[dim] == 0:
            # TODO may need to do subpixel correction for when the dimension has an even length
            psf1d = psf1d * 0
            # psf1d[{dim:np.mean(psf1d.coords[dim])}] = 1
            psf1d[{dim:len(psf1d.coords[dim])/2}] = 1
        else:
            psf1d = psf1d * gaussian(psf1d.coords[dim],np.mean(psf1d.coords[dim]),sigmas[dim])
        
        psf = psf * psf1d
    return psf