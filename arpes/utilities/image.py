import warnings

import numpy as np
import xarray as xr

__all__ = ('imread_to_xarray', 'imread',)


def imread(str_or_path):
    """
    A wrapper around `opencv.imread` and `imageio.imread` that

    1. Falls back to the first available option on the system
    2. Normalizes OpenCV images to RGB format
    3. Removes the alpha channel from imageio.imread data

    :param str_or_path: pathlib.Path or str containing the image to be read
    :return:
    """
    try:
        import cv2
        using_cv2 = True
    except ImportError:
        try:
            import imageio

            using_cv2 = False
        except ImportError as e:
            warnings.warn('You need OpenCV or imageio in order to read images in PyARPES.')
            raise e

    if using_cv2:
        arr = cv2.imread(str(str_or_path))
        return np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1)
    else:
        arr = imageio.imread(str(str_or_path))
        return arr[:,:,:3]


def imread_to_xarray(str_or_path):
    """
    Like `imread`, except that this function wraps the result into a xr.DataArray
    that has x (pixel), y (pixel), and color (['R', 'G', 'B']) dimensions.

    :param str_or_path:
    :return:
    """
    raw_arr = imread(str_or_path)
    sx, sy, _ = raw_arr.shape

    return xr.DataArray(
        raw_arr,
        coords={
            'x': np.asarray(range(sx)),
            'y': np.asarray(range(sy)),
            'color': ['r', 'g', 'b']
        },
        dims=['x', 'y', 'color']
    )
