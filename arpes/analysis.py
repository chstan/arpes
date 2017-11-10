import copy
import functools
import itertools
import warnings

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.spatial import distance

import arpes.models.band
import arpes.utilities
import arpes.utilities.math
from arpes.provenance import provenance, update_provenance

__all__ = ['curvature', 'd1_along_axis', 'd2_along_axis', 'dn_along_axis',
           'gaussian_filter_arr', 'boxcar_filter_arr', 'normalize_by_fermi_distribution',
           'boxcar_filter', 'gaussian_filter', 'fit_bands']


def curvature(arr: xr.DataArray, directions=None, alpha=1, beta=None):
    """
    Defined via
        C(x,y) = ([C_0 + (df/dx)^2]d^2f/dy^2 - 2 * df/dx df/dy d^2f/dxdy + [C_0 + (df/dy)^2]d^2f/dx^2) /
                 (C_0 (df/dx)^2 + (df/dy)^2)^(3/2)

    of in the case of inequivalent dimensions x and y

        C(x,y) = ([1 + C_x(df/dx)^2]C_y * d^2f/dy^2 -
                  2 * C_x * C_y * df/dx df/dy d^2f/dxdy +
                  [1 + C_y * (df/dy)^2] * C_x * d^2f/dx^2) /
                 (1 + C_x (df/dx)^2 + C_y (df/dy)^2)^(3/2)

        where
        C_x = C_y * (xi / eta)^2
        and where (xi / eta) = dx / dy

        The value of C_y can reasonably be taken to have the value |df/dx|_max^2 + |df/dy|_max^2
        C_y = (dy / dx) * (|df/dx|_max^2 + |df/dy|_max^2) * \alpha

        for some dimensionless parameter alpha
    :param arr:
    :param alpha: regulation parameter, chosen semi-universally, but with no particular justification
    :return:
    """
    if beta is not None:
        alpha = np.power(10., beta)

    if directions is None:
        directions = arr.dims[:2]

    axis_indices = tuple(arr.dims.index(d) for d in directions)
    dx, dy = tuple(float(arr.coords[d][1] - arr.coords[d][0]) for d in directions)
    dfx, dfy = np.gradient(arr.values, dx, dy, axis=axis_indices)
    np.nan_to_num(dfx, copy=False)
    np.nan_to_num(dfy, copy=False)

    mdfdx, mdfdy = np.max(np.abs(dfx)), np.max(np.abs(dfy))

    cy = (dy / dx) * (mdfdx ** 2 + mdfdy ** 2) * alpha
    cx = (dx / dy) * (mdfdx ** 2 + mdfdy ** 2) * alpha

    dfx_2, dfy_2 = np.power(dfx, 2), np.power(dfy, 2)
    d2fy = np.gradient(dfy, dy, axis=axis_indices[1])
    d2fx = np.gradient(dfx, dx, axis=axis_indices[0])
    d2fxy = np.gradient(dfx, dy, axis=axis_indices[1])

    denom = np.power((1 + cx * dfx_2 + cy * dfy_2), 1.5)
    numerator = (1 + cx * dfx_2) * cy * d2fy - 2 * cx * cy * dfx * dfy * d2fxy + \
                (1 + cy * dfy_2) * cx * d2fx

    curv = xr.DataArray(
        numerator / denom,
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    del curv.attrs['id']
    provenance(curv, arr, {
        'what': 'Curvature',
        'by': 'curvature',
        'directions': directions,
        'alpha': alpha,
    })
    return curv


def dn_along_axis(arr: xr.DataArray, axis=None, smooth_fn=None, order=2):
    """
    Like curvature, performs a second derivative. You can pass a function to use for smoothing through
    the parameter smooth_fn, otherwise no smoothing will be performed.

    You can specify the axis to take the derivative along with the axis param, which expects a string.
    If no axis is provided the axis will be chosen from among the available ones according to the preference
    for axes here, the first available being taken:

    ['eV', 'kp', 'kx', 'kz', 'ky', 'phi', 'polar']
    :param arr:
    :param axis:
    :param smooth_fn:
    :param order: Specifies how many derivatives to take
    :return:
    """
    axis_order = ['eV', 'kp', 'kx', 'kz', 'ky', 'phi', 'polar']
    if axis is None:
        axes = [a for a in axis_order if a in arr.dims]
        if len(axes):
            axis = axes[0]
        else:
            # have to do something
            axis = arr.dims[0]
            warnings.warn('Choosing axis: {} for the second derivative, no preferred axis found.'.format(axis))

    if smooth_fn is None:
        smooth_fn = lambda x: x

    d_axis = float(arr.coords[axis][1] - arr.coords[axis][0])
    axis_idx = arr.dims.index(axis)

    values = arr.values
    for _ in range(order):
        values = np.gradient(smooth_fn(arr.values), d_axis, axis=axis_idx)

    dn_arr = xr.DataArray(
        values,
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    del dn_arr.attrs['id']
    provenance(dn_arr, arr, {
        'what': '{}th derivative'.format(order),
        'by': 'dn_along_axis',
        'axis': axis,
        'order': order,
    })

    return dn_arr


d2_along_axis = functools.partial(dn_along_axis, order=2)
d1_along_axis = functools.partial(dn_along_axis, order=1)


def unpack_bands_from_fit(band_results: xr.DataArray, weights=None, use_stderr_weighting=True):
    """
    This function is used to deconvolve the band identities of a series of overlapping bands.
    Sometimes through the fitting process, or across a place in the band structure where there is a nodal
    point, the identities of the bands across sequential fits can get mixed up.

    We can try to restore this identity by using the cosine similarity of fits, where the fit is represented
    as a vector by:

        v_band =  (sigma, amplitude, center) * weights
        weights = (5, 1/5, 10)

    For any point in the band structure, we find the closest place where we have fixed the band identities.
    Let the bands be indexed by i so that the bands are b_i and b_i_0 at the point of interest and at the reference
    respectively.

    Then, we calculate the matrix:
        s_ij = sim(b_i, b_j_0)

    The band identities are subsequently chosen so that the trace of this matrix is maximized among possible ways of
    labelling the bands b_i.

    The value of the weights parameter is chosen only to scale the dimensions so that they are closer to the
    same magnitude.

    :param arr:
    :param band_results:
    :param weights:
    :param use_stderr_weighting: Flag to indicate whether to scale vectors by the uncertainty
    :return:
    """
    if weights is None:
        weights = (2, 0, 10,)

    template_components = band_results.values[0].model.components
    prefixes = [component.prefix for component in template_components]

    identified_band_results = copy.deepcopy(band_results)

    def as_vector(model_fit, prefix=''):
        stderr = np.array([model_fit.params[prefix + 'sigma'].stderr,
                           model_fit.params[prefix + 'amplitude'].stderr,
                           model_fit.params[prefix + 'center'].stderr])
        return np.array([model_fit.params[prefix + 'sigma'].value,
                         model_fit.params[prefix + 'amplitude'].value,
                         model_fit.params[prefix + 'center'].value]) * weights / (1 + stderr)

    identified_by_coordinate = {}
    first_coordinate = None
    for coordinate, fit_result in arpes.utilities.enumerate_dataarray(band_results):
        frozen_coord = tuple(coordinate[d] for d in band_results.dims)

        closest_identified = None
        dist = float('inf')
        for coord, identified_band in identified_by_coordinate.items():
            current_dist = np.dot(coord, frozen_coord)
            if current_dist < dist:
                closest_identified = identified_band
                dist = current_dist

        if closest_identified is None:
            first_coordinate = coordinate
            closest_identified = [c.prefix for c in fit_result.model.components], fit_result
            identified_by_coordinate[frozen_coord] = closest_identified

        closest_prefixes, closest_fit = closest_identified
        mat_shape = (len(prefixes), len(prefixes),)
        dist_mat = np.zeros(shape=mat_shape)

        for i, j in np.ndindex(mat_shape):
            dist_mat[i, j] = distance.euclidean(
                as_vector(fit_result, prefixes[i]),
                as_vector(closest_fit, closest_prefixes[j])
            )

        best_arrangement = None
        best_trace = float('inf')
        for p in itertools.permutations(range(len(prefixes))):
            trace = sum(dist_mat[i, p_i] for i, p_i in enumerate(p))
            if trace < best_trace:
                best_trace = trace
                best_arrangement = p

        ordered_prefixes = [closest_prefixes[p_i] for p_i in best_arrangement]
        identified_by_coordinate[frozen_coord] = ordered_prefixes, fit_result
        identified_band_results.loc[coordinate] = ordered_prefixes

    # Now that we have identified the bands,
    # extract them into real bands
    bands = []
    for i in range(len(prefixes)):
        label = identified_band_results.loc[first_coordinate].values.item()[i]
        def dataarray_for_value(param_name, is_value):
            values = np.ndarray(shape=identified_band_results.values.shape, dtype=np.float)
            it = np.nditer(values, flags=['multi_index'], op_flags=['writeonly'])
            while not it.finished:
                prefix = identified_band_results.values[it.multi_index][i]
                param = band_results.values[it.multi_index].params[prefix + param_name]
                if is_value:
                    it[0] = param.value
                else:
                    it[0] = param.stderr
                it.iternext()

            return xr.DataArray(
                values,
                identified_band_results.coords,
                identified_band_results.dims
            )
        band_data = xr.Dataset({
            'center': dataarray_for_value('center', True),
            'center_stderr': dataarray_for_value('center', False),
            'amplitude': dataarray_for_value('amplitude', True),
            'amplitude_stderr': dataarray_for_value('amplitude', False),
            'sigma': dataarray_for_value('sigma', True),
            'sigma_stderr': dataarray_for_value('sigma', False),
        })
        bands.append(arpes.models.band.Band(label, data=band_data))

    return bands


def fit_bands(arr: xr.DataArray, band_description, background=None, direction='mdc',
              preferred_k_direction=None, step=None):
    """
    Fits bands and determines dispersion in some region of a spectrum
    :param arr:
    :param band_description: A description of the bands to fit in the region
    :param background:
    :param direction:
    :return:
    """
    assert(direction in ['edc', 'mdc'])

    def iterate_marginals(arr: xr.DataArray, iterate_directions=None):
        if iterate_directions is None:
            iterate_directions = list(arr.dims)
            iterate_directions.remove('eV')

        selectors = itertools.product(*[arr.coords[d] for d in iterate_directions])
        for ss in selectors:
            coords = dict(zip(iterate_directions, [float(s) for s in ss]))
            yield arr.sel(**coords), coords

    directions = list(tuple(arr.dims))

    broadcast_direction = 'eV'
    if direction == 'mdc':
        if preferred_k_direction is None:
            possible_directions = set(directions).intersection({'kp', 'kx', 'ky'})
            broadcast_direction = list(possible_directions)[0]

    directions.remove(broadcast_direction)

    residual, _ = next(iterate_marginals(arr, directions))
    residual = residual - np.min(residual.values)

    # Let the first band be given by fitting the raw data to this band
    # Find subsequent peaks by fitting models to the residuals
    raw_bands = [band.get('band') if isinstance(band, dict) else band for band in band_description]
    initial_fits = None
    all_fit_parameters = {}

    if step == 'initial':
        residual.plot()

    for band in band_description:
        if isinstance(band, dict):
            band_inst = band.get('band')
            constraints = band.get('constraints', {})
        else:
            band_inst = band
            constraints = None
        fit_model = band_inst.fit_cls(prefix=band_inst.label)
        initial_fit = fit_model.guess_fit(residual, params=constraints)
        if initial_fits is None:
            initial_fits = initial_fit.params
        else:
            initial_fits.update(initial_fit.params)

        residual = residual - initial_fit.best_fit
        if isinstance(band_inst, arpes.models.band.BackgroundBand):
            # This is an approximation to simulate a constant background band underneath the data
            # Because backgrounds are added to our model only after the initial sequence of fits.
            # This is by no means the most appropriate way to do this, just one that works
            # alright for now
            residual = residual - residual.min()

        if step == 'initial':
            residual.plot()
            (residual - residual + initial_fit.best_fit).plot()

    template = arr.sum(broadcast_direction)
    band_results = xr.DataArray(
        np.ndarray(shape=template.values.shape, dtype=object),
        coords=template.coords,
        dims=template.dims,
        attrs=template.attrs
    )

    for marginal, coordinate in iterate_marginals(arr, directions):
        # Use the closest parameters that have been successfully fit, or use the initial
        # parameters, this should be good enough because the order of the iterator will
        # be stable
        closest_model_params = initial_fits # fix me
        distance = float('inf')
        frozen_coordinate = tuple(coordinate[k] for k in template.dims)
        for c, v in all_fit_parameters.items():
            delta = np.array(c) - frozen_coordinate
            current_distance = delta.dot(delta)
            if current_distance < distance:
                current_distance = distance
                closest_model_params = v

        closest_model_params = copy.deepcopy(closest_model_params)

        # TODO mix in any constraints to the model params

        # populate models
        internal_models = [band.fit_cls(prefix=band.label) for band in raw_bands]
        composite_model = functools.reduce(lambda x, y: x + y, internal_models)
        new_params = composite_model.make_params(**{k: v.value for k, v in closest_model_params.items()})
        fit_result = composite_model.fit(marginal.values, new_params,
                                         x=marginal.coords[list(marginal.indexes)[0]].values)

        # insert fit into the results, insert the parameters into the cache so that we have
        # fitting parameters for the next sequence
        band_results.loc[coordinate] = fit_result
        all_fit_parameters[frozen_coordinate] = fit_result.params


    # Unpack the band results
    unpacked_bands = [] #unpack_bands_from_fit(band_results)
    residual = None

    return band_results, unpacked_bands, residual


def gaussian_filter_arr(arr: xr.DataArray, sigma=None, n=1, default_size=1):
    if sigma is None:
        sigma = {}

    sigma = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in sigma.items()}
    for dim in arr.dims:
        if dim not in sigma:
            sigma[dim] = default_size

    sigma = tuple(sigma[k] for k in arr.dims)

    values = arr.values
    for i in range(n):
        values = ndimage.filters.gaussian_filter(values, sigma)

    filtered_arr = xr.DataArray(
        values,
        arr.coords,
        arr.dims,
        attrs=copy.deepcopy(arr.attrs)
    )

    if 'id' in filtered_arr.attrs:
        del filtered_arr.attrs['id']

        provenance(filtered_arr, arr, {
            'what': 'Gaussian filtered data',
            'by': 'gaussian_filter_arr',
            'sigma': sigma,
        })

    return filtered_arr


def gaussian_filter(sigma=None, n=1):
    def f(arr):
        return gaussian_filter_arr(arr, sigma, n)

    return f


def boxcar_filter(size=None, n=1):
    def f(arr):
        return boxcar_filter_arr(arr, size, n)

    return f

def boxcar_filter_arr(arr: xr.DataArray, size=None, n=1, default_size=1, skip_nan=True):
    if size is None:
        size = {}

    size = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in size.items()}
    for dim in arr.dims:
        if dim not in size:
            size[dim] = default_size

    size = tuple(size[k] for k in arr.dims)

    if skip_nan:
        nan_mask = np.copy(arr.values) * 0 + 1
        nan_mask[arr.values != arr.values] = 0
        filtered_mask = ndimage.filters.uniform_filter(nan_mask, size)

        values = np.copy(arr.values)
        values[values != values] = 0

        for i in range(n):
            values = ndimage.filters.uniform_filter(values, size) / filtered_mask
            values[nan_mask == 0] = 0
    else:
        for i in range(n):
            values = ndimage.filters.uniform_filter(values, size)


    filtered_arr = xr.DataArray(
        values,
        arr.coords,
        arr.dims,
        attrs=copy.deepcopy(arr.attrs)
    )

    del filtered_arr.attrs['id']

    provenance(filtered_arr, arr, {
        'what': 'Boxcar filtered data',
        'by': 'boxcar_filter_arr',
        'size': size,
        'skip_nan': skip_nan,
    })

    return filtered_arr

@update_provenance('Normalized by the 1/Fermi Dirac Distribution at sample temp')
def normalize_by_fermi_distribution(data, max_gain=None, rigid_shift=0, instrumental_broadening=None):
    """
    Normalizes a scan by 1/the fermi dirac distribution. You can control the maximum gain with ``clamp``, and whether
    the Fermi edge needs to be shifted (this is for those desperate situations where you want something that
    "just works") via ``rigid_shift``.

    :param data: Input
    :param clamp: Maximum value for the gain. By default the value used is the mean of the spectrum.
    :param rigid_shift: How much to shift the spectrum chemical potential.
    Pass the nominal value for the chemical potential in the scan. I.e. if the chemical potential is at BE=0.1, pass
    rigid_shift=0.1.
    :param instrumental_broadening: Instrumental broadening to use for convolving the distribution
    :return: Normalized DataArray
    """
    distrib = arpes.utilities.math.fermi_distribution(data.coords['eV'].values - rigid_shift, data.S.temp)

    # don't boost by more than 90th percentile of input, by default
    if max_gain is None:
        max_gain = np.mean(data.values)

    distrib[distrib < 1/max_gain] = 1/max_gain
    distrib_arr = xr.DataArray(
        distrib,
        {'eV': data.coords['eV'].values},
        ['eV']
    )

    if instrumental_broadening is not None:
        distrib_arr = gaussian_filter_arr(distrib_arr, sigma={'eV': instrumental_broadening})

    return data / distrib_arr