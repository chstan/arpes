import pytest
import xarray as xr

import arpes.xarray_extensions


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
            'expected': {'dims': ['pixel', 'eV'],
                         'coords': {'pixel': [127, 366, 1], 'eV': [-0.42558, 0.130235, 0.0023256]}},
        }),
        ('main_chamber_load_map', {
            'dataset': 'basic',
            'id': 1,
            'expected': {'dims': ['polar', 'pixel', 'eV'],
                         'coords': {'polar': [-18, 35, 1.9629], 'pixel': [87, 494, 1],
                                    'eV': [-1.369977, 0.431409, 0.0046189]}},
        }),
        ('main_chamber_load_multi_region', {
            'dataset': 'basic',
            'id': 2,
            'expected': {'dims': ['pixel', 'eV'],
                         'coords': {'pixel': [127, 366, 1], 'eV': [-0.42558, 0.130235, 0.0023256]}},
        }),
        ('main_chamber_load_single_cycle', {
            'dataset': 'basic',
            'id': 3,
            'expected': {'dims': ['pixel', 'eV'],
                         'coords': {'pixel': [127, 366, 1], 'eV': [-0.42558, 0.130235, 0.0023256]}},
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
                         'coords': {'phi': [-0.29103, 0.34335, 0.00081749], 'eV': [-2.5, 0.2001, 0.002]}},
        }),
        ('merlin_load_xps', {
            'dataset': 'basic',
            'id': 9,
            'expected': {'dims': ['eV'],
                         'coords': {'eV': [-55, 0.99915, 0.0999]}},
        }),
        ('merlin_load_map', {
            'dataset': 'basic',
            'id': 10,
            'expected': {'dims': ['polar', 'eV', 'phi'],
                         'coords': {'polar': [-0.209439, -0.200713, 0.008726], 'phi': [-0.29103, 0.34335, 0.00081749],
                                    'eV': [-1.33713, 0.33715, 0.00159]}},
        }),
        ('merlin_load_hv', {
            'dataset': 'basic',
            'id': 11,
            'expected': {'dims': ['hv', 'eV', 'phi'],
                         'coords': {'hv': [108, 110, 2], 'phi': [-0.29103, 0.34335, 0.00081749],
                                    'eV': [-1.33911, 0.34312, 0.00159]}},
        }),

        # ALS Beamline 7 "MAESTRO"
        ('maestro_load_cut', {
            'dataset': 'basic',
            'id': 12,
            'expected': {'dims': ['Y', 'X', 'eV'],
                         'coords': {'Y': [4.961, 5.7618, 0.04], 'X': [0.86896, 1.6689, 0.04],
                                    'eV': [-35.478, -31.2837, 0.00805]}},
        }),
        ('maestro_load_xps', {
            'dataset': 'basic',
            'id': 13,
            'expected': {'dims': ['Y', 'X', 'eV'],
                         'coords': {'Y': [-0.92712, -0.777122, 0.010714], 'X': [0.42983, 0.57983, 0.010714],
                                    'eV': [32.389, 39.9296, 0.011272]}},
        }),
        ('maestro_load_map', {
            'dataset': 'basic',
            'id': 14,
            'expected': {'dims': ['Y', 'X', 'eV'],
                         'coords': {'Y': [4.961, 5.7618, 0.04], 'X': [0.86896, 1.6689, 0.04],
                                    'eV': [-35.478, -31.2837, 0.00805]}},
        }),
        ('maestro_load_hv', {
            'dataset': 'basic',
            'id': 15,
            'expected': {'dims': ['Y', 'X', 'eV'],
                         'coords': {'Y': [4.961, 5.7618, 0.04], 'X': [0.86896, 1.6689, 0.04],
                                    'eV': [-35.478, -31.2837, 0.00805]}},
        }),
        ('maestro_load_multi_region', {
            'dataset': 'basic',
            'id': 16,
            'expected': {'dims': ['eV'],
                         'coords': {'eV': [-1.5, 0.50644, 0.0228]}},
        }),
    ]

    def test_load_file_and_basic_attributes(self, sandbox_configuration, dataset, id, expected):
        data = sandbox_configuration.load(dataset, id)
        assert isinstance(data, xr.Dataset)

        # assert basic dataset attributes
        for attr in ['location']: # TODO add spectrum type requirement
            assert attr in data.attrs

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

    def test_momentum_space_conversion(self, sandbox_configuration, dataset, id, expected):
        pass