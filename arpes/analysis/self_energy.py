import xarray as xr
import lmfit as lf
import numpy as np

from arpes.constants import METERS_PER_SECOND_PER_EV_ANGSTROM, HBAR_PER_EV

from arpes.fits.utilities import broadcast_model
from arpes.fits.fit_models import AffineBackgroundModel, LorentzianModel, LinearModel

from typing import Optional, Union

__all__ = ('to_self_energy', 'fit_for_self_energy', 'estimate_bare_band', 'quasiparticle_lifetime',)


BareBandType = Union[xr.DataArray, str, lf.model.ModelResult]
DispersionType = Union[xr.DataArray, xr.Dataset]


def get_peak_parameter(data: xr.DataArray, parameter_name: str) -> xr.DataArray:
    """
    Extracts a parameter from a potentially prefixed peak-like component in a simple model or composite model

    :param data:
    :param parameter_name:
    :return:
    """
    first_item = data.values.ravel()[0]
    peak_like = (lf.models.LorentzianModel, lf.models.VoigtModel, lf.models.GaussianModel, lf.models.PseudoVoigtModel)

    if isinstance(first_item, lf.model.ModelResult):
        if isinstance(first_item.model, lf.model.CompositeModel):
            peak_like_components = [c for c in first_item.model.components if isinstance(c, peak_like)]
            assert len(peak_like_components) == 1

            return data.F.p('{}{}'.format(peak_like_components[0].prefix, parameter_name))
        else:
            return data.F.p(parameter_name)

    raise ValueError('Unsupported dispersion {}, expected xr.DataArray[lmfit.ModelResult]'.format(
        type(data)))


def local_fermi_velocity(bare_band: xr.DataArray):
    """
    Calculates the band velocity under assumptions of a linear bare band.
    """

    fitted_model = LinearModel().guess_fit(bare_band)
    raw_velocity = fitted_model.params['slope'].value

    if 'eV' in bare_band.dims:
        # the "y" values are in `bare_band` are momenta and the "x" values are energy, therefore
        # the slope is dy/dx = dk/dE
        raw_velocity = 1 / raw_velocity

    return raw_velocity * METERS_PER_SECOND_PER_EV_ANGSTROM


def estimate_bare_band(dispersion: xr.DataArray, bare_band_specification: Optional[str] = None):
    """
    Estimates the bare band from a fitted dispersion. This can be done in a few ways:

    1. None: Equivalent to 'baseline_linear' below
    2. 'linear': A linear fit to the dispersion is used, and this also provides the fermi_velocity
    3. 'ransac_linear': A linear fit with random sample consensus (RANSAC) region will be used and this
       also provides the fermi_velocity
    4. 'hough': Hough transform based method

    :param dispersion:
    :param bare_band_specification:
    :return:
    """
    try:
        centers = get_peak_parameter(dispersion, 'center')
    except ValueError:
        centers = dispersion

    mom_options = [d for d in dispersion.dims if d in {'k', 'kp', 'kx', 'ky', 'kz'}]
    assert len(mom_options) <= 1
    fit_dimension = 'eV' if 'eV' in dispersion.dims else mom_options[0]

    if bare_band_specification is None:
        bare_band_specification = 'ransac_linear'

    initial_linear_fit = LinearModel().guess_fit(centers)
    if bare_band_specification == 'linear':
        fitted_model = initial_linear_fit
    elif bare_band_specification == 'ransac_linear':
        from skimage.measure import LineModelND, ransac
        min_samples = len(centers.coords[fit_dimension]) // 10
        residual_threshold = np.median(np.abs(initial_linear_fit.residual)) * 1
        ransac_model, inliers = ransac(
            np.stack([centers.coords[fit_dimension], centers]).T, LineModelND,
            max_trials=1000, min_samples=min_samples,
            residual_threshold=residual_threshold
        )
        inlier_data = centers.where(
            xr.DataArray(inliers, coords=dict([[fit_dimension, centers.coords[fit_dimension]]]), dims=[fit_dimension]),
            drop=True)

        fitted_model = LinearModel().guess_fit(inlier_data)
    elif bare_band_specification == 'hough':
        raise NotImplementedError('Hough Transform estimate of bare band not yet supported.')
    else:
        raise ValueError('Unrecognized bare band type: {}'.format(bare_band_specification))

    ys = fitted_model.eval(x=centers.coords[fit_dimension])
    return xr.DataArray(ys, centers.coords, centers.dims)


def quasiparticle_lifetime(self_energy: xr.DataArray, bare_band: xr.DataArray) -> xr.DataArray:
    """
    Calculates the quasiparticle mean free path in meters (meters!). The bare band is used to calculate
    the band/Fermi velocity and internally the procedure to calculate the quasiparticle lifetime is used

    :param self_energy:
    :param bare_band:
    :return:
    """
    imaginary_part = np.abs(np.imag(self_energy)) / 2
    return HBAR_PER_EV / imaginary_part


def quasiparticle_mean_free_path(self_energy: xr.DataArray, bare_band: xr.DataArray) -> xr.DataArray:
    lifetime = quasiparticle_lifetime(self_energy, bare_band)
    return lifetime * local_fermi_velocity(bare_band)


def to_self_energy(dispersion: xr.DataArray, bare_band: Optional[BareBandType] = None, k_independent=True,
                   fermi_velocity=None) -> xr.Dataset:
    """
    Converts MDC fit results into the self energy. This largely consists of extracting
    out the linewidth and the difference between the dispersion and the bare band value.

    lorentzian(x, amplitude, center, sigma) =
        (amplitude / pi) * sigma/(sigma^2 + ((x-center))**2)


    Once we have the curve-fitted dispersion we can calculate the self energy if we also
    know the bare-band dispersion. If the bare band is not known, then at least the imaginary
    part of the self energy is still calculable, and a guess as to the real part can be
    calculated under assumptions of the bare band dispersion as being free electron like wih
    effective mass m* or being Dirac like (these are equivalent at low enough energy).

    Acceptabe bare band spefications are discussed in detail in `estimate_bare_band` above.

    To future readers of the code, please note that the half-width half-max of a Lorentzian is equal to the
    $\gamma$ parameter, which defines the imaginary part of the self energy.

    :param dispersion:
    :param bare_band:
    :param k_independent:
    :param fermi_velocity:
    :return:
    """
    if not k_independent:
        raise NotImplementedError('PyARPES does not currently support self energy analysis '
                                  'except in the k-independent formalism.')

    if isinstance(dispersion, xr.Dataset):
        dispersion = dispersion.results

    from_mdcs = 'eV' in dispersion.dims # if eV is in the dimensions, then we fitted MDCs
    estimated_bare_band = estimate_bare_band(dispersion, bare_band)

    if fermi_velocity is None:
        fermi_velocity = local_fermi_velocity(estimated_bare_band)

    imaginary_part = get_peak_parameter(dispersion, 'fwhm') / 2
    centers = get_peak_parameter(dispersion, 'center')

    if from_mdcs:
        imaginary_part *= fermi_velocity / METERS_PER_SECOND_PER_EV_ANGSTROM
        real_part = -((centers * fermi_velocity / METERS_PER_SECOND_PER_EV_ANGSTROM) - dispersion.coords['eV'].values)
    else:
        real_part = centers - bare_band

    self_energy = xr.DataArray(
        real_part + 1.j * imaginary_part,
        coords=dispersion.coords, dims=dispersion.dims,
    )

    return xr.Dataset({
        'self_energy': self_energy,
        'bare_band': estimated_bare_band,
    })


def fit_for_self_energy(data: xr.DataArray, method='mdc', bare_band: Optional[BareBandType] = None,
                        **kwargs) -> xr.Dataset:
    """
    Fits for the self energy of a dataset containing a single band.

    The bare band shape
    :param data:
    :param method: one of 'mdc' and 'edc'
    :param bare_band:
    :return:
    """

    if method == 'mdc':
        fit_results = broadcast_model([LorentzianModel, AffineBackgroundModel], data, 'eV', **kwargs)
    else:
        possible_mometum_dims = (
            'phi', 'theta', 'psi', 'beta',
            'kp', 'kx', 'ky', 'kz',
        )
        mom_axes = set(data.dims).intersection(possible_mometum_dims)

        if len(mom_axes) > 1:
            raise ValueError('Too many possible momentum dimensions, please clarify.')
        fit_results = broadcast_model([LorentzianModel, AffineBackgroundModel], data, list(mom_axes)[0], **kwargs)

    return to_self_energy(fit_results, bare_band=bare_band)