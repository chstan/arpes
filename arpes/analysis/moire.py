"""
arpes.moire includes some tools for analyzing moirés and data on moiré heterostructures in particular.

All of the moirés discussed here are on hexagonal crystal systems.
"""

import matplotlib.pyplot as plt

from arpes.utilities.bz import hex_cell_2d
from arpes.plotting.bz import bz_plot, Translation, Rotation

import numpy as np
from scipy.spatial.distance import pdist


def higher_order_commensurability(lattice_constant_ratio, order=2, angle_range=None):
    """
    Unfinished

    :param lattice_constant_ratio:
    :param order:
    :param angle_range:
    :return:
    """
    if angle_range is None:
        angle_range = (0, 5)

    bz_corner_a, bz_corner_b = np.asarray([1, 0]), np.asarray([0.5, np.sqrt(3) / 2])

    return bz_corner_a, bz_corner_b


def mod_points_to_lattice(pts, a, b):
    rmat = np.asarray([[0, -1], [1, 0]])
    ra, rb = rmat @ a, rmat @ b
    ra, rb = ra / (np.sqrt(3) / 2), rb / (np.sqrt(3) / 2)

    return (pts +
            np.outer(np.ceil(pts @ rb), a) +
            np.outer(np.ceil(pts @ -ra), b))


def generate_other_lattice_points(a, b, ratio, order=1, angle=0):
    ratio = max(np.abs(ratio), 1 / np.abs(ratio))
    cosa, sina = np.cos(angle), np.sin(angle)
    rmat = np.asarray([[cosa, -sina], [sina, cosa]])
    a, b = rmat @ (ratio * a), rmat @ (ratio * b)

    ias = np.arange(-order, order + 1)
    pts = (a[None, None, :] * ias[None, :, None]) + (b[None, None, :] * ias[:, None, None])
    pts = pts.reshape(len(ias) ** 2, 2)

    # not quite correct, since we need the manhattan distance
    # mask = np.linalg.norm(pts, axis=1) <= order * ratio

    ds = np.stack([(np.outer(ias[None, :], (ias[None, :] * 0 + 1))),
                   (np.outer(ias[None, :] * 0 + 1, (ias[None, :])))], axis=-1).reshape(len(ias) ** 2, 2)

    dabs = np.abs(np.sum(ds, axis=1))
    dist = np.max(np.abs(ds), axis=1)
    sign = np.sign(ds)
    sign = sign[:, 0] == sign[:, 1]
    dist[sign] = dabs[sign]

    return pts[dist <= order]


def unique_points(pts):
    return np.vstack([np.array(u) for u in set([tuple(p) for p in pts])])


def generate_segments(grouped_points, a, b):
    moded = mod_points_to_lattice(grouped_points, a, b)
    g1d = np.diff(np.sum(grouped_points, axis=1))
    m1d = np.diff(np.sum(moded, axis=1))

    low_index = 0
    for split_index in np.nonzero(np.abs((m1d - g1d)) > 1e-11)[0]:
        yield moded[low_index:split_index + 1]
        low_index = split_index + 1

    yield moded[low_index:]


def minimum_distance(pts, a, b):
    moded = np.stack([mod_points_to_lattice(x, a, b) for x in pts], axis=1)
    return np.min(np.stack([pdist(x) for x in moded], axis=-1), axis=0)


def calculate_bz_vertices_from_direct_cell(cell):
    from ase.dft.bz import bz_vertices

    if len(cell) > 2:
        assert all(abs(cell[2][0:2]) < 1e-6) and all(abs(cell.T[2][0:2]) < 1e-6)
    else:
        cell = [list(c) + [0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T
    try:
        bz1 = bz_vertices(icell[:3, :3], dim=2)
    except TypeError:
        bz1 = bz_vertices(icell[:3, :3])

    return bz1


def angle_between_vectors(a, b):
    return np.arccos(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calc_commensurate_moire_cell(underlayer_a, overlayer_a, relative_angle=0, swap_angle=False):
    """
    Calculates nearly commensurate moire unit cells for two hexagonal lattices
    :return:
    """
    from ase.dft.kpoints import get_special_points
    from ase.dft.bz import bz_vertices

    underlayer_direct = hex_cell_2d(a=underlayer_a)
    overlayer_direct = hex_cell_2d(a=overlayer_a)

    underlayer_direct = [list(c) + [0] for c in underlayer_direct] + [[0, 0, 1]]
    overlayer_direct = [list(c) + [0] for c in overlayer_direct] + [[0, 0, 1]]

    underlayer_icell = np.linalg.inv(underlayer_direct).T
    overlayer_icell = np.linalg.inv(overlayer_direct).T

    underlayer_k = np.dot(underlayer_icell.T, get_special_points(underlayer_direct)['K'])
    overlayer_k = Rotation.from_rotvec([0, 0, relative_angle]).apply(
        np.dot(overlayer_icell.T, get_special_points(overlayer_direct)['K']))

    moire_k = (underlayer_k - overlayer_k)
    moire_a = underlayer_a * (np.linalg.norm(underlayer_k) / np.linalg.norm(moire_k))
    moire_angle = angle_between_vectors(underlayer_k, moire_k)

    if swap_angle:
        moire_angle = -moire_angle

    moire_cell = hex_cell_2d(moire_a)
    moire_cell = [list(c) + [0] for c in moire_cell] + [[0, 0, 1]]
    moire_cell = Rotation.from_rotvec([0, 0, moire_angle]).apply(moire_cell)
    moire_icell = np.linalg.inv(moire_cell).T

    moire_bz_points = bz_vertices(moire_icell)
    moire_bz_points = moire_bz_points[[len(p[0]) for p in moire_bz_points].index(6)][0]

    return {
        'k_points': (underlayer_k, overlayer_k, moire_k),
        'moire_a': moire_a,
        'moire_k': moire_k,
        'moire_cell': moire_cell,
        'moire_icell': moire_icell,
        'moire_bz_points': moire_bz_points,
        'moire_bz_angle': moire_angle,
    }


def plot_simple_moire_unit_cell(underlayer_a, overlayer_a, relative_angle, ax=None, offset=True,
                                swap_angle=False):
    """
    Plots a digram of a moiré unit cell.
    :return:
    """

    if ax is None:
        _, ax = plt.subplots()

    bz_plot(cell=hex_cell_2d(a=underlayer_a), linewidth=1, ax=ax, paths=[], hide_ax=False, set_equal_aspect=False)
    bz_plot(cell=hex_cell_2d(a=overlayer_a), linewidth=1, ax=ax, paths=[],
            transformations=[Rotation.from_rotvec([0, 0, relative_angle])], hide_ax=False, set_equal_aspect=False)

    moire_info = calc_commensurate_moire_cell(underlayer_a, overlayer_a, relative_angle, swap_angle=swap_angle)
    moire_k = moire_info['moire_k']

    if offset:
        k_offset = Rotation.from_rotvec([0, 0, 120 * np.pi / 180]).apply(moire_k)
    else:
        k_offset = 0

    bz_plot(cell=hex_cell_2d(a=moire_info['moire_a']), linewidth=1, ax=ax, paths=[],
            transformations=[Rotation.from_rotvec([0, 0, -moire_info['moire_bz_angle']]),
                             Translation(moire_info['k_points'][0] + k_offset)],
            hide_ax=True, set_equal_aspect=True)

