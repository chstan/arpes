import numpy as np
from arpes.typing import DataType
from arpes.typing import xr_types
from arpes.utilities import normalize_to_spectrum

__all__ = ('deconvolve_ice',)

def _convolve(original_data, convolution_kernel):
    conv_kern_norm = convolution_kernel / np.sum(convolution_kernel)
    n_points = min(len(original_data), len(conv_kern_norm))
    padding = np.ones(n_points)
    temp = np.concatenate((padding * original_data[0], original_data, padding * original_data[-1]))
    convolved = np.convolve(temp, conv_kern_norm, mode='valid')
    n_offset = int((len(convolved) - n_points) / 2)
    result = (convolved[n_offset:])[:n_points]
    return result

def deconvolve_ice(data: DataType,psf,n_iterations=5,deg=None):
    """Deconvolves data by a given point spread function.
    
    :param data:
    :param psf:
    :param n_iterations -- the number of convolutions to use for the fit (default 5):
    :param deg -- the degree of the fitting polynominal (default n_iterations-3):
    :return numpy.ndarray:
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
    color_list = np.linspace(0,0.9,n_iterations+1)[1:]

    for i in range(n_iterations-1):
        iteration_list.append(_convolve(iteration_list[-1],psf))
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