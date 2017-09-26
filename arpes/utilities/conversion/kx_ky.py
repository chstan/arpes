import numpy as np
import xarray as xr

FINE_K_GRAINING = 0.01
MEDIUM_K_GRAINING = 0.05
COARSE_K_GRAINING = 0.1

from arpes.utilities.conversion import (
    # photon energy scans
    kx_kz_E_to_polar,
    kx_kz_E_to_hv,
    polar_hv_E_to_kx,
    polar_hv_E_to_kz,
    polar_hv_E_corners_to_kx_kz_E_bounds,
    jacobian_polar_hv_E_to_kx_kz_E,

    # rotation and analyzer deflection scans
    kx_ky_KE_to_polar,
    kx_ky_KE_to_elev,
    polar_elev_KE_to_kx,
    polar_elev_KE_to_ky,
    polar_elev_KE_corners_to_kx_ky_KE_bounds,
    jacobian_polar_elev_KE_to_kx_ky_KE,
)

_coordinate_translations = {
    # Organized by old -> new,
    # which is another way of saying original spectrum axes -> new axes
    (('detector_angle', 'hv', 'KE'),('kx','kz','KE'),): {
        'detector_angle': kx_kz_E_to_polar,
        'hv': kx_kz_E_to_hv,
        'KE': lambda x, y, KE, metadata: KE,
        'kx': polar_hv_E_to_kx,
        'kz': polar_hv_E_to_kz,
        'bounds_from_corners': polar_hv_E_corners_to_kx_kz_E_bounds,
        'jacobian': jacobian_polar_hv_E_to_kx_kz_E,
    },
    # This coordinate translation set is used for the vertical slit orientations
    # with moving analyzers, such as is used at BL10
    (('detector_angle', 'detector_sweep_angle', 'KE'),('kx','ky','KE',)): {
        'detector_angle': kx_ky_KE_to_polar,
        'detector_sweep_angle': kx_ky_KE_to_elev,
        'KE': lambda x, y, KE, metadata: KE,
        'kx': polar_elev_KE_to_kx,
        'ky': polar_elev_KE_to_ky,
        'bounds_from_corners': polar_elev_KE_corners_to_kx_ky_KE_bounds,
        'jacobian': jacobian_polar_elev_KE_to_kx_ky_KE,
    },
}

def convert_to_k_space(arr: xr.DataArray, dk=0.05, dk_z=0.05):
    # TODO Unimplemented
    pass

def convert_to_kx_ky(arr: xr.DataArray, dk=0.05):
    # TODO make this work for DataArrays that have more degrees of freedom than just eV, phi, theta

    kx_min = 0
    ky_min = 0
    kx_max = 0
    ky_max = 0

    kspace_coords = {
        'eV': arr.coords['eV'].values,
        'kx': np.arange(kx_min, kx_max, dk),
        'ky': np.arange(ky_min, ky_max, dk),
    }

    converted_volume = np.zeros((len(kspace_coords['eV']),
                                 len(kspace_coords['kx']),
                                 len(kspace_coords['ky'])))

    Xgs, Ygs, Zgs = np.meshgrid(kspace_coords['eV'], kspace_coords['kx'], kspace_coords['ky'], indexing='ij')
    Xgs, Ygs, Zgs = Xgs.ravel(), Ygs.ravel(), Zgs.ravel()

    data = RegularGridInterpolator(
        points=,
        values=,
        method='linear',
        fill_value=float('nan')
    )

    return xr.DataArray(
        converted_volume,
        kspace_coords,
        ['eV', 'kx', 'ky'],
        attrs=arr.attrs

    )