import numpy as np
import xarray as xr
from arpes.provenance import update_provenance

__all__ = ('infer_center_pixel', 'stitch_maps', )

def infer_center_pixel(arr: xr.DataArray):
    import arpes.xarray_extensions

    edges = arr.S.find_spectrum_angular_edges()
    return float((np.max(edges) + np.min(edges)) / 2 + np.min(arr.coords['pixel']))

@update_provenance('Stitch maps together')
def stitch_maps(arr: xr.DataArray, arr2: xr.DataArray, dimension='beta'):
    """
    Stitches together two maps by appending and potentially dropping frames in the first dataset.

    This is useful for beamline work when the beam is lost or in L-ARPES if laser output is blocked for part
    of a scan and a subsequent scan was taken to repair the problem.
    :param arr: Incomplete map
    :param arr2: completion of first map
    :return:
    """

    # as a first step we need to align the coords of map2 onto those of map1
    coord2 = arr2.coords[dimension].data.copy()
    coord1 = arr.coords[dimension].data.copy()

    first_repair_coordinate = coord2.data[0]
    i, lower, higher = None, None, None

    # search for the breakpoint
    for i, (lower, higher,) in enumerate(zip(coord1, coord1[1:])):
        if higher > first_repair_coordinate:
            break

    delta_low, delta_high = lower - first_repair_coordinate, higher - first_repair_coordinate
    if abs(delta_low) < abs(delta_high):
        delta = delta_low
    else:
        delta = delta_high
        i += 1

    shifted_repair_map = arr2.copy()
    shifted_repair_map.coords[dimension].data += delta

    attrs = arr.attrs.copy()
    good_data_slice = dict()
    good_data_slice[dimension] = slice(None, i)

    selected = arr.isel(**good_data_slice)
    selected.attrs.clear()
    shifted_repair_map.attrs.clear()
    concatted = xr.concat([selected, shifted_repair_map], dim=dimension)
    return xr.DataArray(
        concatted.data,
        concatted.coords,
        concatted.dims,
        attrs=attrs,
    )
