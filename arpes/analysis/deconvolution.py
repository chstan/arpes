import numpy as np

__all__ = ('deconvolve_ice')

def _convolve(original_data, convolution_kernel):
    conv_kern_norm = convolution_kernel / np.sum(convolution_kernel)
    n_points = min(len(original_data), len(conv_kern_norm))
    padding = np.ones(n_points)
    temp = np.concatenate((padding * original_data[0], original_data, padding * original_data[-1]))
    convolved = np.convolve(temp, conv_kern_norm, mode='valid')
    n_offset = int((len(convolved) - n_points) / 2)
    result = (convolved[n_offset:])[:n_points]
    return result

# deconvolution by iterated-convolution extrapolation
def deconvolve_ice(spectrum,psf,n_iterations=5,deg=None):
    if deg is None:
        deg = n_iterations - 3
    iteration_steps = list(range(1,n_iterations+1))

    iteration_list = [spectrum]
    color_list = np.linspace(0,0.9,n_iterations+1)[1:]

    for i in range(n_iterations-1):
        iteration_list.append(convolve(iteration_list[-1],psf))
    iteration_list = np.asarray(iteration_list)

    restoration = spectrum*0
    for t, series in enumerate(iteration_list.T):
        coefs = np.polyfit(iteration_steps,series,deg=n_iterations-3)
        poly = np.poly1d(coefs)
        restoration[t] = poly(0)
        
    return restoration