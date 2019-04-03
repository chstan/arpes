import warnings

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch

from arpes.analysis.mask import apply_mask_to_coords
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.geometry import polyhedron_intersect_plane
from arpes.utilities.bz import build_2dbz_poly, process_kpath
from arpes.plotting.utils import path_for_plot


__all__ = ('annotate_special_paths', 'bz2d_plot', 'bz3d_plot', 'bz_plot',
           'plot_data_to_bz', 'plot_data_to_bz2d', 'plot_data_to_bz3d', 'plot_plane_to_bz',)


def plot_plane_to_bz(cell, plane, ax, special_points=None, facecolor='red'):
    from ase.dft.bz import bz_vertices

    if isinstance(plane, str):
        plane_points = process_kpath(plane, cell, special_points=special_points)[0]
    else:
        plane_points = plane

    d1, d2 = plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]

    faces = [p[0] for p in bz_vertices(np.linalg.inv(cell).T)]
    pts = polyhedron_intersect_plane(faces, np.cross(d1, d2), plane_points[0])

    collection = Poly3DCollection([pts])
    collection.set_facecolor(facecolor)
    ax.add_collection3d(collection, zs='z')


def plot_data_to_bz(data: DataType, cell, **kwargs):
    if len(data) == 3:
        return plot_data_to_bz3d(data, cell, **kwargs)

    return plot_data_to_bz2d(data, cell, **kwargs)


def plot_data_to_bz2d(data: DataType, cell, rotate=None, shift=None, scale=None, ax=None,
                      mask=True, out=None, bz_number=None, **kwargs):
    data = normalize_to_spectrum(data)

    assert('You must k-space convert data before plotting to BZs' and data.S.is_kspace)

    if bz_number is None:
        bz_number = (0,0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,9))
        bz2d_plot(cell, paths='all', ax=ax)

    if len(cell) == 2:
        cell = [list(c) + [0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T

    # Prep coordinates and mask
    raveled = data.T.meshgrid(as_dataset=True)
    dims = data.dims
    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        rotation = np.array([(c, -s), (s, c)])


        raveled = raveled.T.transform_coords(dims, rotation)

    if scale is not None:
        raveled = raveled.T.scale_coords(dims, scale)

    if shift is not None:
        raveled = raveled.T.shift_coords(dims, shift)

    copied = data.values.copy()

    if mask:
        built_mask = apply_mask_to_coords(raveled, build_2dbz_poly(cell=cell), dims)
        copied[built_mask.T] = np.nan

    cmap = kwargs.get('cmap', matplotlib.cm.Blues)
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    cmap.set_bad((1, 1, 1, 0))

    delta_x = np.dot(np.array(bz_number), icell[:2, 0])
    delta_y = np.dot(np.array(bz_number), icell[:2, 1])

    ax.pcolormesh(raveled.data_vars[dims[0]].values + delta_x, raveled.data_vars[dims[1]].values + delta_y, copied.T, cmap=cmap)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax



def plot_data_to_bz3d(data: DataType, cell, **kwargs):
    raise NotImplementedError('plot_data_to_bz3d is not implemented yet.')


def bz_plot(cell, *args, **kwargs):
    if len(cell) > 2:
        return bz3d_plot(cell, *args, **kwargs)

    return bz2d_plot(cell, *args, **kwargs)


def bz3d_plot(cell, vectors=False, paths=None, points=None, ax=None,
              elev=None, scale=1, repeat=None, offset=None,
              hide_ax=True, **kwargs):
    """
    For now this is lifted from ase.dft.bz.bz3d_plot with some modifications. All copyright and
    licensing terms for this and bz2d_plot are those of the current release of ASE
    (Atomic Simulation Environment).

    :param cell:
    :param vectors:
    :param paths:
    :param points:
    :param elev:
    :param scale:
    :return:
    """
    try:
        import ase
    except ImportError:
        warnings.warn('You will need to install ASE (Atomic Simulation Environment) to use this feature.')
        raise ImportError('You will need to install ASE before using Brillouin Zone plotting')

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    icell = np.linalg.inv(cell).T
    kpoints = points

    if isinstance(paths, str):
        from ase.dft.kpoints import (get_special_points, special_paths,
                                     parse_path_string, get_cellinfo,
                                     crystal_structure_from_cell)
        special_points = get_special_points(cell)
        structure = crystal_structure_from_cell(cell)

        path_string = special_paths[structure] if paths == 'all' else paths
        paths = []
        for names in parse_path_string(path_string):
            points = []
            for name in names:
                points.append(np.dot(icell.T, special_points[name]))
                paths.append((names, points))

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca(projection='3d')

    azim = np.pi / 5
    elev = elev or np.pi / 6
    x = np.sin(azim)
    y = np.cos(azim)
    view = [x * np.cos(elev), y * np.cos(elev), np.sin(elev)]

    from ase.dft.bz import bz_vertices # dynamic because we do not require ase
    bz1 = bz_vertices(icell)

    maxp = 0.0

    if repeat is None:
        repeat = (1, 1, 1,)

    dx, dy, dz = icell[0], icell[1], icell[2]
    rep_x, rep_y, rep_z = repeat

    if isinstance(rep_x, int):
        rep_x = (0, rep_x)
    if isinstance(rep_y, int):
        rep_y = (0, rep_y)
    if isinstance(rep_z, int):
        rep_z = (0, rep_z)

    c = kwargs.pop('c', 'k')
    c = kwargs.pop('color', c)


    for ix, iy, iz in itertools.product(range(*rep_x), range(*rep_y), range(*rep_z)):
        delta = dx * ix + dy * iy + dz * iz

        for points, normal in bz1:
            color = c

            if np.dot(normal, view) < 0:
                ls = ':'
            else:
                ls = '-'

            cosines = np.dot(icell, normal)/ np.linalg.norm(normal) / np.linalg.norm(icell, axis=1)
            for idx, cosine in enumerate(cosines):
                if np.abs(np.abs(cosine) - 1) < 1e-6 and False: # debugging this
                    tup = [rep_x, rep_y, rep_z][idx]
                    current = [ix, iy, iz][idx]

                    if cosine < 0:
                        current = current - 1

                    if tup[0] < current + 1 < tup[1]:
                        color = (1, 1, 1, 0)

                    if current + 1 != tup[1] and current != tup[0]:
                        ls = ':'
                        color='blue'

            x, y, z = np.concatenate([points, points[:1]]).T
            x, y, z = x + delta[0], y + delta[1], z + delta[2]

            ax.plot(x, y, z, c=color, ls=ls, **kwargs)
            maxp = max(maxp, points.max())

    if vectors:
        ax.add_artist(Arrow3D([0, icell[0, 0]],
                              [0, icell[0, 1]],
                              [0, icell[0, 2]],
                              mutation_scale=20, lw=1,
                              arrowstyle='-|>', color='k'))
        ax.add_artist(Arrow3D([0, icell[1, 0]],
                              [0, icell[1, 1]],
                              [0, icell[1, 2]],
                              mutation_scale=20, lw=1,
                              arrowstyle='-|>', color='k'))
        ax.add_artist(Arrow3D([0, icell[2, 0]],
                              [0, icell[2, 1]],
                              [0, icell[2, 2]],
                              mutation_scale=20, lw=1,
                              arrowstyle='-|>', color='k'))
        maxp = max(maxp, 0.6 * icell.max())

    if paths is not None:
        for names, points in paths:
            x, y, z = np.array(points).T
            ax.plot(x, y, z, c='r', ls='-')

            for name, point in zip(names, points):
                x, y, z = point
                if name == 'G':
                    name = '\\Gamma'
                elif len(name) > 1:
                    name = name[0] + '_' + name[1]
                ax.text(x, y, z, '$' + name + '$',
                        ha='center', va='bottom', color='r')

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], p[1], p[2], c='b')

    if hide_ax:
        ax.set_axis_off()
        ax.autoscale_view(tight=True)

    s = maxp / 0.5 * 0.45 * scale
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_zlim(-s, s)
    ax.set_aspect('equal')

    ax.view_init(azim=azim / np.pi * 180, elev=elev / np.pi * 180)


def annotate_special_paths(ax, paths, cell=None, offset=None, special_points=None, labels=None, **kwargs):
    if paths == '':
        raise ValueError('Must provide a proper path.')

    if isinstance(paths, (list, str)):
        if isinstance(paths, str):
            if labels is None:
                labels = paths

            paths = process_kpath(paths, cell, special_points=special_points)

            if not isinstance(labels[0], list):
                labels = [labels]

            labels = [list(l) for l in labels]
            paths = list(zip(labels, paths))


    fontsize = kwargs.pop('fontsize', 14)
    if offset is None:
        offset = dict()

    two_d = True
    try:
        ax.get_zlim()
        two_d = False
    except AttributeError:
        pass

    for names, points in paths:
        x, y, z = np.array(points).T

        if two_d:
            ax.plot(x, y, c='r', ls='-', **kwargs)
        else:
            ax.plot(x, y, z, c='r', ls='-', **kwargs)

        for name, point in zip(names, points):
            x, y, z = point
            display_name = name
            if display_name == 'G':
                display_name = '\\Gamma'
            elif len(name) > 1:
                name = name[0] + '_' + name[1]

            off = offset.get(name, 0)
            try:
                if two_d:
                    x, y = x + off[0], y + off[1]
                else:
                    x, y, z = x + off[0], y + off[1], z + off[2]

            except TypeError:
                if two_d:
                    x, y = x + off, y + off
                else:
                    x, y, z = x + off, y + off, z + off

            if two_d:
                ax.text(x, y, '$' + display_name + '$', ha='center', va='bottom', color='r', fontsize=fontsize)
            else:
                ax.text(x, y, z, '$' + display_name + '$', ha='center', va='bottom', color='r', fontsize=fontsize)


def bz2d_plot(cell, vectors=False, paths=None, points=None, repeat=None, ax=None,
              offset=None, hide_ax=True, **kwargs):
    """
    This piece of code modified from ase.ase.dft.bz.py:bz2d_plot and follows copyright and license for ASE.

    Plots a Brillouin zone corresponding to a given unit cell
    :param cell:
    :param vectors:
    :param paths:
    :param points:
    :return:
    """
    import matplotlib.pyplot as plt
    from ase.dft.bz import bz_vertices

    # 2d in x-y plane
    if len(cell) > 2:
        assert all(abs(cell[2][0:2]) < 1e-6) and all(abs(cell.T[2][0:2]) < 1e-6)
    else:
        cell = [list(c) + [0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T
    kpoints = points

    if ax is None:
        ax = plt.axes()

    bz1 = bz_vertices(icell)

    if isinstance(paths, str):
        from ase.dft.kpoints import (get_special_points, special_paths,
                                     parse_path_string, get_cellinfo,
                                     crystal_structure_from_cell)
        special_points = get_special_points(cell)
        structure = crystal_structure_from_cell(cell)

        path_string = special_paths[structure] if paths == 'all' else paths
        paths = []
        for names in parse_path_string(path_string):
            points = []
            for name in names:
                points.append(np.dot(icell.T, special_points[name]))
                paths.append((names, points))

    maxp = 0.0
    c = kwargs.pop('c', 'k')
    c = kwargs.pop('color', c)

    for points, normal in bz1:
        x, y, z = np.concatenate([points, points[:1]]).T
        ax.plot(x, y, c=c, ls='-', **kwargs)
        maxp = max(maxp, points.max())

    if repeat is not None:
        dx, dy = icell[0], icell[1]

        rep_x, rep_y = repeat
        if isinstance(rep_x, int):
            rep_x = (0, rep_x)
        if isinstance(rep_y, int):
            rep_y = (0, rep_y)

        for ix, iy in itertools.product(range(*rep_x), range(*rep_y)):
            print(ix, iy)
            delta = dx * ix + dy * iy

            for points, normal in bz1:
                x, y, z = np.concatenate([points, points[:1]]).T
                x, y = x + delta[0], y + delta[1]

                c = kwargs.pop('c', 'k')
                c = kwargs.pop('color', c)
                ax.plot(x, y, c=c, ls='-', **kwargs)
                maxp = max(maxp, points.max())

    if vectors:
        ax.arrow(0, 0, icell[0, 0], icell[0, 1],
                 lw=1, color='k',
                 length_includes_head=True,
                 head_width=0.03, head_length=0.05)
        ax.arrow(0, 0, icell[1, 0], icell[1, 1],
                 lw=1, color='k',
                 length_includes_head=True,
                 head_width=0.03, head_length=0.05)
        maxp = max(maxp, icell.max())

    if paths is not None:
        annotate_special_paths(ax, paths, offset=offset)

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], p[1], c='b')

    if hide_ax:
        ax.set_axis_off()
        ax.autoscale_view(tight=True)

    s = maxp * 1.05

    ax.set_aspect('equal')

