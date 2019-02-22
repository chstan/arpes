import numpy as np
from arpes.typing import DataType
from arpes.typing import xr_types
from arpes.utilities import normalize_to_spectrum

import scipy
from skimage import restoration

__all__ = ('deconvolve_ice','deconvolve_rl')

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

def deconvolve_rl(data: DataType,psf,n_iterations=10,axis=None,mode='reflect'):
    """Deconvolves data by a given point spread function using the Richardson-Lucy method.
    
    :param data:
    :param psf:
    :param n_iterations -- the number of convolutions to use for the fit (default 50):
    :param axis:
    :return DataArray or numpy.ndarray -- based on input type:
    """
    
    arr = normalize_to_spectrum(data)
    
    if type(arr) is not np.ndarray:
        arr = arr.values

    if axis is None:
        u = [arr]

        for i in range(n_iterations):
            c = scipy.ndimage.convolve(u[-1],psf,mode=mode)
            u.append(u[-1] * scipy.ndimage.convolve(arr/c,psf,mode=mode))

        result = u[-1]
    elif len(data.dims) > 1:
        raise NotImplementedError
        """
        # choose axis to convolve for 1D convolution
        if axis is not None:
            if axis not in data.dims:
                # problem!
                raise KeyError
        elif 'eV' in data.dims:
            axis = 'eV'
        else:
            axis = data.dims[0]

        result = normalize_to_spectrum(data).copy(deep=True)

        # not sure this is the best way to do this
        if len(data.dims) == 2:
            axis_index = list(data.dims).index(axis)
            axis_other = list(data.dims)[1-axis_index]

            if axis_index == 0:
                new_arr = arr.copy()
                for i, (coord,edc) in enumerate(data.T.iterate_axis(axis_other)):
                    new_arr[i] = deconvolve_rl(edc.spectrum.values,psf=psf,n_iterations=n_iterations)
                result.values = new_arr
            elif axis_index == 1:
                new_arr = arr.copy().T
                for i, (coord,edc) in enumerate(data.T.iterate_axis(axis_other)):
                    new_arr[i] = deconvolve_rl(edc.spectrum.values,psf=psf,n_iterations=n_iterations)
                result.values = new_arr.T
        """
    else:
        u = [arr]

        for i in range(n_iterations):
            c = scipy.ndimage.convolve(u[-1],psf)
            u.append(u[-1] * scipy.ndimage.convolve(arr/c,np.flip(psf)))  # not yet tested to ensure flip correct for asymmetric psf

        if type(data) is np.ndarray:
            result = u[-1].copy()
        else:
            result = normalize_to_spectrum(data).copy(deep=True)
            result.values = u[-1]
            
    return result