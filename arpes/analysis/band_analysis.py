import copy
import functools
import itertools

import numpy as np
import xarray as xr
from scipy.spatial import distance

import arpes.models.band
import arpes.utilities.math
from arpes.utilities import enumerate_dataarray, normalize_to_spectrum
from arpes.utilities.jupyter_utils import wrap_tqdm
from arpes.typing import DataType
from arpes.constants import HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ
from arpes.fits import broadcast_model, LorentzianModel, AffineBackgroundModel, QuadraticModel
from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward

__all__ = ('fit_bands', 'fit_for_effective_mass',)


def fit_for_effective_mass(data: DataType, fit_kwargs=None):
    """
    We should probably include uncertainties here.

    :param data:
    :param fit_kwargs:
    :return:
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    data = normalize_to_spectrum(data)
    mom_dim = [d for d in ['kp', 'kx', 'ky', 'kz', 'phi', 'beta', 'theta'] if d in data.dims][0]

    results = broadcast_model([LorentzianModel, AffineBackgroundModel], data, mom_dim, **fit_kwargs)
    if mom_dim in {'phi', 'beta', 'theta'}:
        forward = convert_coordinates_to_kspace_forward(data)
        final_mom = [d for d in ['kx', 'ky', 'kp', 'kz'] if d in forward][0]
        eVs = results.F.p('a_center').values
        kps = [forward[final_mom].sel(eV=eV, **dict([[mom_dim, ang]]), method='nearest') for
               eV, ang in zip(eVs, data.coords[mom_dim].values)]
        quad_fit = QuadraticModel().fit(eVs, x=np.array(kps))

        return HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ / (2 * quad_fit.params['a'].value)

    quad_fit = QuadraticModel().guess_fit(results.F.p('a_center'))
    return HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ / (2 * quad_fit.params['a'].value)


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
    for coordinate, fit_result in enumerate_dataarray(band_results):
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


def fit_patterned_bands(arr: xr.DataArray, band_set, direction_normal=True,
                        fit_direction=None, avoid_crossings=None,
                        stray=None, background=True, preferred_k_direction=None,
                        interactive=True, dataset=True):
    """
    Fits bands and determines dispersion in some region of a spectrum.

    The dimensions of the dataset are partitioned into three types:

    1. Fit directions, these are coordinates along the 1D (or maybe later 2D) marginals
    2. Broadcast directions, these are directions used to interpolate against the patterned directions
    3. Free directions, these are broadcasted but they are not used to extract initial values of the fit parameters

    For instance, if you laid out band patterns in a E, k_p, delay spectrum at delta_t=0, then if you are using MDCs,
    k_p is the fit direction, E is the broadcast direction, and delay is a free direction.

    In general we can recover the free directions and the broadcast directions implicitly by examining the band_set
    passed as a pattern.

    :param arr:
    :param band_set: dictionary with bands and points along the spectrum
    :param orientation: edc or mdc
    :param direction_normal:
    :param preferred_k_direction:
    :param dataset:
    :return: Dataset or DataArray, as controlled by the parameter "dataset"
    """

    if background == True:
        from arpes.models.band import AffineBackgroundBand
        background = AffineBackgroundBand

    free_directions = list(arr.dims)
    free_directions.remove(fit_direction)

    def is_between(x, y0, y1):
        y0, y1 = np.min([y0, y1]), np.max([y0, y1])
        return y0 <= x <= y1

    def interpolate_itersecting_fragments(coord, coord_index, points):
        """
        Finds all consecutive pairs of points in `points`
        :param coord:
        :param coord_idx:
        :param points:
        :return:
        """

        assert(len(points[0]) == 2) # only support 2D interpolation

        for point_low, point_high in zip(points, points[1:]):
            coord_other_index = 1 - coord_index

            check_coord_low, check_coord_high = point_low[coord_index], point_high[coord_index]
            if is_between(coord, check_coord_low, check_coord_high):
                # this is unnecessarily complicated
                if check_coord_low < check_coord_high:
                    yield coord, (coord - check_coord_low) / (check_coord_high - check_coord_low) * \
                          (point_high[coord_other_index] - point_low[coord_other_index]) + \
                          point_low[coord_other_index]
                else:
                    yield coord, (coord - check_coord_high) / (check_coord_low - check_coord_high) * \
                          (point_low[coord_other_index] - point_high[coord_other_index]) + \
                          point_high[coord_other_index]


    def resolve_partial_bands_from_description(
            coord_dict, name=None, band=arpes.models.band.Band, dims=None,
            params=None, points=None, marginal=None):
        # You don't need to supply a marginal, but it is useful because it allows estimation of the initial value for
        # the amplitude from the approximate peak location

        if params is None:
            params = {}


        coord_name = [d for d in dims if d in coord_dict][0]
        iter_coord_value = coord_dict[coord_name]
        partial_band_locations = list(interpolate_itersecting_fragments(
           iter_coord_value, arr.dims.index(coord_name), points or []))

        def build_params(old_params, center, center_stray=None):
            new_params = copy.deepcopy(old_params)
            new_params.update({
                'center': { 'value': center, }
            })
            if center_stray is not None:
                new_params['center']['min'] = center - center_stray
                new_params['center']['max'] = center + center_stray
                new_params['sigma'] = new_params.get('sigma', {})
                new_params['sigma']['value'] = center_stray
                if marginal is not None:
                    near_center = marginal.sel(**dict([[marginal.dims[0], slice(
                        center - 1.2 * center_stray,
                        center + 1.2 * center_stray,
                    )]]))

                    low, high = np.percentile(near_center.values, (20, 80,))
                    new_params['amplitude'] = new_params.get('amplitude', {})
                    new_params['amplitude']['value'] = high - low
            return new_params

        return [{
            'band': band,
            'name': '{}_{}'.format(name, i),
            'params': build_params(params, band_center, params.get('stray', stray)), # TODO
        } for i, (_, band_center) in enumerate(partial_band_locations)]

    template = arr.sum(fit_direction)
    band_results = xr.DataArray(
        np.ndarray(shape=template.values.shape, dtype=object),
        coords=template.coords,
        dims=template.dims,
        attrs=template.attrs
    )

    total_slices = np.product([len(arr.coords[d]) for d in free_directions])
    for coord_dict, marginal in wrap_tqdm(arr.T.iterate_axis(free_directions), interactive,
                                          desc='fitting',
                                          total=total_slices):
        partial_bands = [resolve_partial_bands_from_description(
            coord_dict, marginal=marginal, **b) for b in band_set.values()]

        partial_bands = [p for p in partial_bands if len(p)]

        if background is not None and len(partial_bands):
            partial_bands = partial_bands + [[{
                'band': background,
                'name': '',
                'params': {},
            }]]

        def instantiate_band(partial_band):
            phony_band = partial_band['band'](partial_band['name'])
            built = phony_band.fit_cls(prefix=partial_band['name'], missing='drop')
            for constraint_coord, params in partial_band['params'].items():
                if constraint_coord == 'stray':
                    continue
                built.set_param_hint(constraint_coord, **params)
            return built

        internal_models = [instantiate_band(b) for bs in partial_bands for b in bs]

        if len(internal_models) == 0:
            band_results.loc[coord_dict] = None
            continue

        composite_model = functools.reduce(lambda x, y: x + y, internal_models)
        new_params = composite_model.make_params()
        fit_result = composite_model.fit(marginal.values, new_params,
                                         x=marginal.coords[list(marginal.indexes)[0]].values)

        # populate models, sample code
        band_results.loc[coord_dict] = fit_result

    if not dataset:
        band_results.attrs['original_data'] = arr
        return band_results

    residual = arr.copy(deep=True)
    residual.values = np.zeros(residual.shape)

    for coords in band_results.T.iter_coords():
        fit_item = band_results.sel(**coords).item()
        if fit_item is None:
            continue

        try:
            residual.loc[coords] = fit_item.residual
        except:
            pass

    return xr.Dataset({
        'data': arr,
        'residual': residual,
        'results': band_results,
        'norm_residual': residual / arr,
    }, residual.coords)


def fit_bands(arr: xr.DataArray, band_description, background=None,
              direction='mdc', preferred_k_direction=None, step=None):
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

    directions = list(arr.dims)

    broadcast_direction = 'eV'

    if direction == 'mdc':
        if preferred_k_direction is None:
            possible_directions = set(directions).intersection({'kp', 'kx', 'ky', 'phi'})
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
            params = band.get('params', {})
        else:
            band_inst = band
            params = None
        fit_model = band_inst.fit_cls(prefix=band_inst.label)
        initial_fit = fit_model.guess_fit(residual, params=params)
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
            pass
            #residual = residual - residual.min()

        if step == 'initial':
            residual.plot()
            (residual - residual + initial_fit.best_fit).plot()

    if step == 'initial':
        return None, None, residual

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
                if direction == 'mdc': # TODO remove me
                    closest_model_params = v

        closest_model_params = copy.deepcopy(closest_model_params)

        # TODO mix in any params to the model params

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
    #unpacked_bands = unpack_bands_from_fit(band_results)
    unpacked_bands = None
    residual = None

    return band_results, unpacked_bands, residual

