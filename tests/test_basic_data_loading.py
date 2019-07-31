import pytest
import numpy as np
import xarray as xr

import arpes.xarray_extensions
from arpes.utilities.conversion import convert_to_kspace


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []

    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append(([x[1] for x in items]))

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope='class')


class TestBasicDataLoading(object):
    """
    Tests procedures/plugins for loading basic data. This is a bit gross because of how we are
    parameterizing the test, ideally we would pass kwargs and then we can hide what we don't need.
    """

    data = None

    scenarios = [
        # Lanzara Group "Main Chamber"
        ('main_chamber_load_cut', {
            'dataset': 'basic',
            'id': 0,
            'expected': {'dims': ['phi', 'eV'],
                         'coords': {'phi': [0.22165, 0.63879, 0.001745], 'eV': [-0.42558, 0.130235, 0.0023256],
                                    'alpha': 0, 'x': -0.770435, 'y': 34.75, 'z': -3.4e-5, 'chi': -6.25 * np.pi / 180},
                         'offset_coords': {'phi': 0.22165 - 0.405, 'beta': 0, 'chi': 0,}},
        }),
        ('main_chamber_load_map', {
            'dataset': 'basic',
            'id': 1,
            'expected': {'dims': ['beta', 'phi', 'eV'],
                         'coords': {'beta': [-0.314159, 0.61086, 0.03426], 'phi': [0.15184, 0.862193, 0.001745],
                                    'eV': [-1.369977, 0.431409, 0.0046189], 'x': 2.99, 'y': 41.017, 'z': -0.03104,
                                    'alpha': 0,},
                         'offset_coords': {'phi': 0.15184 - 0.405, 'beta': -0.314159, 'chi': 0,}},
        }),
        ('main_chamber_load_multi_region', {
            'dataset': 'basic',
            'id': 2,
            'expected': {'dims': ['phi', 'eV'],
                         'coords': {'phi': [0.22165, 0.63879, 0.001745], 'eV': [-0.42558, 0.130235, 0.0023256],
                                    'alpha': 0,},
                         'offset_coords': {'phi': 0.22165 - 0.405, 'beta': 0, 'chi': 0,}},
        }),
        ('main_chamber_load_single_cycle', {
            'dataset': 'basic',
            'id': 3,
            'expected': {'dims': ['phi', 'eV'],
                         'coords': {'phi': [0.22165, 0.63879, 0.001745], 'eV': [-0.42558, 0.130235, 0.0023256],
                                    'alpha': 0,},
                         'offset_coords': {'phi': 0.22165 - 0.405, 'beta': 0, 'chi': 0,}},
        }),

        # Lanzara Group "Spin-ToF"
        # ('stof_load_edc', {
        #     'dataset': 'basic',
        #     'id': 4,
        #     'expected': {},
        # }),
        # ('stof_load_spin_edc', {
        #     'dataset': 'basic',
        #     'id': 5,
        #     'expected': {},
        # }),
        # ('stof_load_map', {
        #     'dataset': 'basic',
        #     'id': 6,
        #     'expected': {},
        # }),
        # ('stof_load_spin_map', {
        #     'dataset': 'basic',
        #     'id': 7,
        #     'expected': {},
        # }),

        # ALS Beamline 4 "MERLIN" / SES
        ('merlin_load_cut', {
            'dataset': 'basic',
            'id': 8,
            'expected': {'dims': ['eV', 'phi'],
                         'coords': {'phi': [-0.29103, 0.34335, 0.00081749], 'eV': [-2.5, 0.2001, 0.002],
                                    'alpha': np.pi / 2,},
                         'offset_coords': {'phi': -0.29103, 'theta': 0.1043, 'chi': 0}},
        }),
        ('merlin_load_xps', {
            'dataset': 'basic',
            'id': 9,
            'expected': {'dims': ['eV'],
                         'coords': {'eV': [-55, 0.99915, 0.0999],
                                    'alpha': np.pi / 2,
                                    'chi': -107.09 * np.pi / 180},
                         'offset_coords': {'phi': 0, 'theta': 0.002 * np.pi / 180, 'chi': 0}},
        }),
        ('merlin_load_map', {
            'dataset': 'basic',
            'id': 10,
            'expected': {'dims': ['theta', 'eV', 'phi'],
                         'coords': {'theta': [-0.209439, -0.200713, 0.008726], 'phi': [-0.29103, 0.34335, 0.00081749],
                                    'eV': [-1.33713, 0.33715, 0.00159],
                                    'alpha': np.pi / 2},
                         'offset_coords': {'phi': -0.29103, 'theta': -0.209439, 'chi': 0}},
        }),
        ('merlin_load_hv', {
            'dataset': 'basic',
            'id': 11,
            'expected': {'dims': ['hv', 'eV', 'phi'],
                         'coords': {'hv': [108, 110, 2], 'phi': [-0.29103, 0.34335, 0.00081749],
                                    'eV': [-1.33911, 0.34312, 0.00159],
                                    'alpha': np.pi / 2,},
                         'offset_coords': {'phi': -0.29103, 'theta': -0.999 * np.pi / 180, 'chi': 0}},
        }),

        # ALS Beamline 7 "MAESTRO"
        ('maestro_load_cut', {
            'dataset': 'basic',
            'id': 12,
            'expected': {'dims': ['y', 'x', 'eV'],
                         'coords': {'y': [4.961, 5.7618, 0.04], 'x': [0.86896, 1.6689, 0.04],
                                    'eV': [-35.478, -31.2837, 0.00805],
                                    'z': -0.4, 'alpha': 0},
                         'offset_coords': {'phi': 0, 'theta': 0, 'chi': 0}},
        }),
        ('maestro_load_xps', {
            'dataset': 'basic',
            'id': 13,
            'expected': {'dims': ['y', 'x', 'eV'],
                         'coords': {'y': [-0.92712, -0.777122, 0.010714], 'x': [0.42983, 0.57983, 0.010714],
                                    'eV': [32.389, 39.9296, 0.011272],
                                    'alpha': 0},
                         'offset_coords': {'phi': 0, 'theta': 10.008 * np.pi / 180, 'chi': 0,}},
        }),
        ('maestro_load_map', {
            'dataset': 'basic',
            'id': 14,
            'expected': {'dims': ['y', 'x', 'eV'],
                         'coords': {'y': [4.961, 5.7618, 0.04], 'x': [0.86896, 1.6689, 0.04],
                                    'eV': [-35.478, -31.2837, 0.00805],
                                    'alpha': 0},
                         'offset_coords': {'phi': 0, 'theta': 0, 'chi': 0,}},
        }),
        ('maestro_load_hv', {
            'dataset': 'basic',
            'id': 15,
            'expected': {'dims': ['y', 'x', 'eV'],
                         'coords': {'y': [4.961, 5.7618, 0.04], 'x': [0.86896, 1.6689, 0.04],
                                    'eV': [-35.478, -31.2837, 0.00805],
                                    'alpha': 0},
                         'offset_coords': {'phi': 0, 'theta': 0, 'chi': 0,}},
        }),
        ('maestro_load_multi_region', {
            'dataset': 'basic',
            'id': 16,
            'expected': {'dims': ['eV'],
                         'coords': {'eV': [-1.5, 0.50644, 0.0228],
                                    'alpha': 0},
                         'offset_coords': {'phi': 0, 'theta': 10.062 * np.pi / 180, 'chi': 0,}},
        }),
    ]

    def test_load_file_and_basic_attributes(self, sandbox_configuration, dataset, id, expected):
        data = sandbox_configuration.load(dataset, id)
        assert isinstance(data, xr.Dataset)

        # assert basic dataset attributes
        for attr in ['location']: # TODO add spectrum type requirement
            assert attr in data.attrs

        # assert that all necessary coordinates are present
        necessary_coords = {'phi', 'psi', 'alpha', 'chi', 'beta', 'theta', 'x', 'y', 'z', 'hv'}
        for necessary_coord in necessary_coords:
            assert necessary_coord in data.coords

        # assert basic spectrum attributes
        for attr in ['hv', 'location']:
            if attr == 'hv' and (data.S.spectrum.attrs.get('spectrum_type') == 'hv_map' or
                                 data.S.is_multi_region):
                continue
            assert attr in data.S.spectrum.attrs

        # assert dimensions
        assert list(data.S.spectra[0].dims) == expected['dims']

        # assert coordinate shape
        by_dims = data.S.spectra[0].dims
        ranges = [[pytest.approx(data.coords[d].min().item(), 1e-3),
                   pytest.approx(data.coords[d].max().item(), 1e-3),
                   pytest.approx(data.T.stride(generic_dim_names=False)[d], 1e-3)] for d in by_dims]

        assert list(zip(by_dims, ranges)) == list(zip(by_dims, [expected['coords'][d] for d in by_dims]))
        for k, v in expected['coords'].items():
            if isinstance(v, float):
                assert k and (pytest.approx(data.coords[k].item(), 1e-3) == v)

        def safefirst(x):
            try:
                x = x[0]
            except (TypeError, IndexError):
                pass

            try:
                x = x.item()
            except AttributeError:
                pass

            return x

        for k in expected['offset_coords'].keys():
            offset = safefirst(data.S.spectra[0].S.lookup_offset_coord(k))
            assert k and (pytest.approx(offset, 1e-3) == expected['offset_coords'][k])

        kspace_data = convert_to_kspace(data)
        assert(isinstance(kspace_data, xr.DataArray))