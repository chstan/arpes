import numpy as np
import xarray as xr
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.conversion import slice_along_path
from arpes.fits.fit_models import LorentzianModel, AffineBackgroundModel
from sklearn.decomposition import PCA

__all__ = ('curves_along_pocket', 'edcs_along_pocket', 'radial_edcs_along_pocket',
           'pocket_parameters',)


def pocket_parameters(data: DataType, kf_method=None, sel=None, method_kwargs=None, **kwargs):
    """
    Estimates pocket center, anisotropy, principal vectors, and extent in either
    angle or k-space. Since data can be converted forward it is generally advised to do
    this analysis in angle space before conversion.
    :param data:
    :param kf_method:
    :param sel:
    :param method_kwargs:
    :param kwargs:
    :return:
    """
    slices, angles = curves_along_pocket(data, **kwargs)

    if kf_method is None:
        kf_method = find_kf_by_mdc

    if sel is None:
        sel = {'eV': slice(-0.03, 0.05)}

    kfs = [kf_method(s if sel is None else s.sel(**sel), **(method_kwargs or {})) for s in slices]

    fs_dims = list(data.dims)
    if 'eV' in fs_dims:
        fs_dims.remove('eV')

    locations = [{d: ss[d].sel(angle=kf, eV=0, method='nearest').item() for d in fs_dims} for kf, ss in
                 zip(kfs, slices)]

    location_vectors = [[coord[d] for d in fs_dims] for coord in locations]
    as_ndarray = np.array(location_vectors)

    pca = PCA(n_components=2)
    pca.fit(as_ndarray)

    return {
        'locations': locations,
        'location_vectors': location_vectors,
        'center': {d: np.mean(np.array([coord[d] for coord in locations])) for d in fs_dims},
        'pca': pca.components_,
    }


def radial_edcs_along_pocket(data: DataType, angle, inner_radius=0, outer_radius=5,
                             n_points=None, select_radius=None, **kwargs):
    """
    Produces EDCs distributed radially along a vector from the pocket center. The pocket center
    should be passed through kwargs via `{dim}={value}`. I.e. an appropriate call would be

    radial_edcs_along_pocket(spectrum, np.pi / 4, inner_radius=1, outer_radius=4, phi=0.1, beta=0)

    :param data: ARPES Spectrum
    :param angle: Angle along the FS to cut against
    :param n_points: Number of EDCs, can be automatically inferred
    :param select_radius:
    :param kwargs:
    :return:
    """

    data = normalize_to_spectrum(data)
    fermi_surface_dims = list(data.dims)

    assert('eV' in fermi_surface_dims)
    fermi_surface_dims.remove('eV')

    center_point = {k: v for k, v in kwargs.items() if k in data.dims}
    center_as_vector = np.array([center_point.get(d, 0) for d in fermi_surface_dims])

    if n_points is None:
        stride = data.T.stride(generic_dim_names=False)
        granularity = np.mean(np.array([stride[d] for d in fermi_surface_dims]))
        n_points = int(1. * (outer_radius - inner_radius) / granularity)

    if n_points <= 0:
        n_points = 10

    primitive = np.array([np.cos(angle), np.sin(angle)])
    far = center_as_vector + outer_radius * primitive
    near = center_as_vector + inner_radius * primitive
    vecs = zip(near, far)

    radius_coord = np.linspace(inner_radius, outer_radius, n_points or 10)

    data_vars = {}
    for d, points in dict(zip(fermi_surface_dims, vecs)).items():
        print(d, points)
        data_vars[d] = xr.DataArray(
            np.array(np.linspace(points[0], points[1], n_points)),
            coords={'r': radius_coord}, dims=['r']
        )

    selection_coords = [{k: v[n] for k, v in data_vars.items()} for n in range(n_points)]

    edcs = [data.S.select_around(coord, radius=select_radius, fast=True)
            for coord in selection_coords]

    for r, edc in zip(radius_coord, edcs):
        edc.coords['r'] = r

    data_vars['data'] = xr.concat(edcs, dim='r')

    return xr.Dataset(data_vars, coords=data_vars['data'].coords)


def curves_along_pocket(data: DataType, n_points=None, inner_radius=0, outer_radius=5, shape=None,
                        **kwargs):
    """
    Produces radial slices along a Fermi surface through a pocket. Evenly distributes perpendicular cuts along an
    ellipsoid. The major axes of the ellipsoid can be specified by `shape` but must be axis aligned.

    The inner and outer radius parameters control the endpoints of the resultant slices along the Fermi surface
    :param data:
    :param n_points:
    :param inner_radius:
    :param outer_radius:
    :param shape:
    :param kwargs:
    :return:
    """
    data = normalize_to_spectrum(data)

    fermi_surface_dims = list(data.dims)
    if 'eV' in fermi_surface_dims:
        fermi_surface_dims.remove('eV')

    center_point = {k: v for k, v in kwargs.items() if k in data.dims}

    center_as_vector = np.array([center_point.get(d, 0) for d in fermi_surface_dims])

    if n_points is None:
        # determine N approximately by the granularity
        n_points = np.min(list(len(data.coords[d].values) for d in fermi_surface_dims))

    stride = data.T.stride(generic_dim_names=False)
    resolution = np.min([v for s, v in stride.items() if s in fermi_surface_dims])

    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    def slice_at_angle(theta):
        primitive = np.array([np.cos(theta), np.sin(theta)])
        far = center_as_vector + outer_radius * primitive

        return slice_along_path(data, [dict(zip(fermi_surface_dims, point))
                                       for point in [center_as_vector, far]], resolution=resolution)

    slices = [slice_at_angle(theta) for theta in angles]

    max_ang = slices[0].coords['angle'].max().item()

    slices = [s.sel(angle=slice(max_ang * (1.0 * inner_radius / outer_radius), None)).isel(angle=slice(None, -1))
              for s in slices]

    for ang, s in zip(angles, slices):
        s.coords['theta'] = ang

    # we do not xr.concat because the interpolated angular dim can actually be different on each due
    # to floating point nonsense
    return slices, angles


def find_kf_by_mdc(slice: DataType, offset=0, **kwargs):
    """
    Offset is used to control the radial offset from the pocket for studies where
    you want to go slightly off the Fermi momentum
    :param slice:
    :param offset:
    :param kwargs:
    :return:
    """
    if isinstance(slice, xr.Dataset):
        slice = slice.data

    assert(isinstance(slice, xr.DataArray))

    if 'eV' in slice.dims:
        slice = slice.sum('eV')

    lor = LorentzianModel()
    bkg = AffineBackgroundModel(prefix='b_')

    result = (lor + bkg).guess_fit(data=slice, params=kwargs)
    return result.params['center'].value + offset


def edcs_along_pocket(data: DataType, kf_method=None, select_radius=None,
                      sel=None, method_kwargs=None, **kwargs):
    slices, angles = curves_along_pocket(data, **kwargs)

    if kf_method is None:
        kf_method = find_kf_by_mdc

    if sel is None:
        sel = {'eV': slice(-0.05, 0.05)}

    kfs = [kf_method(s if sel is None else s.sel(**sel), **(method_kwargs or {})) for s in slices]

    fs_dims = list(data.dims)
    if 'eV' in fs_dims:
        fs_dims.remove('eV')

    locations = [{d: ss[d].sel(angle=kf, eV=0, method='nearest').item() for d in fs_dims} for kf, ss in
                 zip(kfs, slices)]

    edcs = [data.S.select_around(l, radius=select_radius, fast=True)
            for l in locations]

    data_vars = {}
    index = np.array(angles)

    for d in fs_dims:
        data_vars[d] = xr.DataArray(
            np.array([l[d] for l in locations]), coords={'theta': index}, dims=['theta'])

    for ang, edc in zip(angles, edcs):
        edc.coords['theta'] = ang

    data_vars['spectrum'] = xr.concat(edcs, dim='theta')

    return xr.Dataset(data_vars, coords=data_vars['spectrum'].coords)
