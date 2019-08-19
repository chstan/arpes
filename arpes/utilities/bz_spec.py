"""
Contains specifications and paths to BZs for some common materials. These are
useful if you know them ahead of hand, either from a picture, from a DFT
calculation, tight binding calculation, or explicit specification.

This is used in the interactive BZ explorer in order to help with orienting
yourself in momentum.

Each zone definition corresponds to the following things:

1. The geometry, this gives the physical Brillouin zone size and shape
2. The material work function, if available
3. The material inner potential, if available
4. The material name
"""

import functools
import numpy as np
import pathlib

from arpes.utilities.bz import hex_cell_2d, as_2d

A_GRAPHENE = 2.46 / (2 * np.pi)
A_WS2 = 3.15 / (2 * np.pi)
A_WSe2 = 3.297 / (2 * np.pi)


def bz_points_for_hexagonal_lattice(a=1):
    from ase.dft.bz import bz_vertices

    cell = hex_cell_2d(a)
    cell = [list(c) + [0] for c in cell] + [[0, 0, 1]]
    icell = np.linalg.inv(cell).T
    bz_vertices = bz_vertices(icell)

    # get the first face which has six points, this is the top or bottom
    # face of the cell
    return as_2d(bz_vertices[[len(face[0]) for face in bz_vertices].index(6)][0])


def image_for(file):
    f = pathlib.Path(__file__).parent / '..' / 'example_data' / 'brillouin_zones' / file
    return str(f.absolute())


SURFACE_ZONE_DEFINITIONS = { # : Dict[str, Dict[str, any]]
    '2H-WS2': {
        'name': '2H-Tungsten Disulfide',
        'work_function': None,
        'inner_potential': None,
        'bz_points': functools.partial(bz_points_for_hexagonal_lattice, a=A_WS2),
    },
    'Graphene': {
        'name': 'Graphene',
        'work_function': None,
        'inner_potential': None,
        'bz_points': functools.partial(bz_points_for_hexagonal_lattice, a=A_GRAPHENE),
    },
    '2H-WSe2': {
        'name': 'Tungsten Diselenide',
        'work_function': None,
        'inner_potential': None,
        'bz_points': functools.partial(bz_points_for_hexagonal_lattice, a=A_WS2),
    },
    '1T-TiSe2': {
        'name': '1T-Titanium Diselenide',
        'work_function': None,
        'inner_potential': None,
        'image': image_for('1t-tise2-bz.png'),
        'image_waypoints': [
            # everywhere waypoints are pixel_x, pixel_y, mom_x, mom_y
            # two waypoints are requried in order to specify
            [],
            [],
        ],
        'image_src': 'https://arxiv.org/abs/1712.04967',
    },
    'Td-WTe2': {
        'name': 'Td-Tungsten Ditelluride',
        'work_function': None,
        'inner_potential': None,
        'image': image_for('td-wte2-bz.png'),
        'image_waypoints': [
            [445, 650, -0.4, -0.2],
            [1470, 166, 0.4, 0.2],
        ],
        'image_src': 'https://arxiv.org/abs/1603.08508',
    },
    'NCCO': {
        'name': 'Nd_{2-x}Ce_xCuO_4',
        'work_function': None,
        'inner_potential': None,
        'image': image_for('cuprate-bz.png'),
        'image_waypoints': [
            [],
            [],
        ],
        'image_src': 'https://vishiklab.faculty.ucdavis.edu/wp-content/uploads/sites/394/2016/12/ARPES-studies-of-cuprates-online.pdf',
    },
    'Bi2212': {
        'name': 'Bi_2Sr_2CaCu_2O_{8+x}',
        'work_function': None,
        'inner_potential': None,
        'image': image_for('cuprate-bz.png'),
        'image_waypoints': [
            [],
            [],
        ],
        'image_src': 'https://vishiklab.faculty.ucdavis.edu/wp-content/uploads/sites/394/2016/12/ARPES-studies-of-cuprates-online.pdf',
    },
    '1H-NbSe2': {
        'name': '1H-Niobium Diselenide',
        'work_function': None,
        'inner_potential': None,
        'image': image_for('1h-nbse2-bz.png'),
        'image_waypoints': [
            [],
            [],
        ],
        'image_src': 'https://www.nature.com/articles/s41467-018-03888-4',
    },
    '1H-TaS2': {
        'name': '1H-Tantalum Disulfide',
        'work_function': None,
        'inner_potential': None,
        'image': image_for('1h-tas2-bz.png'),
        'image_waypoints': [
            [],
            [],
        ],
        'image_src': 'https://www.nature.com/articles/s41467-018-03888-4',
    },
}
