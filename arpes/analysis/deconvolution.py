import numpy as np

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

def deconvolve_ice(spectrum,psf,n_iterations=5,deg=None):
    """Deconvolve a spectrum by a given point spread function.
    
    :param spectrum -- numpy.ndarray:
    :param psf:
    :param n_iterations -- the number of convolutions to use for the fit (default 5):
    :param deg -- the degree of the fitting polynominal (default n_iterations-3):
    :return numpy.ndarray:
    """
    
    if deg is None:
        deg = n_iterations - 3
    iteration_steps = list(range(1,n_iterations+1))

    iteration_list = [spectrum]
    color_list = np.linspace(0,0.9,n_iterations+1)[1:]

    for i in range(n_iterations-1):
        iteration_list.append(_convolve(iteration_list[-1],psf))
    iteration_list = np.asarray(iteration_list)

    restoration = spectrum*0
    for t, series in enumerate(iteration_list.T):
        coefs = np.polyfit(iteration_steps,series,deg=n_iterations-3)
        poly = np.poly1d(coefs)
        restoration[t] = poly(0)
        
    return restoration